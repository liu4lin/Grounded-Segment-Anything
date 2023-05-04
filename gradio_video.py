import os
import random

import gradio as gr
import argparse

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import numpy as np

# diffusers
import torch
from diffusers import StableDiffusionInpaintPipeline

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

import glob


def generate_caption(processor, blip_model, raw_image):
    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def transform_image(image_pil):

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def draw_box(box, draw, label):
    # random color
    color = tuple(np.random.randint(0, 255, size=3).tolist())

    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color,  width=2)

    if label:
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white")

        draw.text((box[0], box[1]), label)



config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "/checkpoints/GroundingDINO/groundingdino_swint_ogc.pth"
sam_checkpoint = '/checkpoints/SAM/sam_vit_h_4b8939.pth'
output_dir="outputs"
device="cuda"


blip_processor = None
blip_model = None
groundingdino_model = None
sam_predictor = None
import cv2
import time
import uuid

def run_bg_change_ffmpeg(input_video, bg_image, text_prompt, box_threshold, text_threshold, filter_size):
    #input_video '/tmp/d17aa7fa3ea36a39fac6aa64da457a2899ad3abc/1.mp4':
    #base_dir = os.path.join("/tmp", str(uuid.uuid4()))
    base_dir = os.path.dirname(input_video)
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')
    output_file = os.path.join(base_dir, 'output.mp4')

    # Ensure output directory exists
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build the FFmpeg command
    ffmpeg_cmd = "/usr/bin/ffmpeg -i {} -vf \"fps=5\" {}/%04d.png".format(input_video, input_dir)
    os.system(ffmpeg_cmd)
    frame_pths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    counter = 0
    for frame_pth in frame_pths:
        in_image = Image.open(frame_pth)
        output, _ = run_grounded_sam(in_image, bg_image, text_prompt, box_threshold, text_threshold, filter_size)
        output_pth = os.path.join(output_dir, frame_pth.split('/')[-1])
        output.save(output_pth)
        print("No. of frames:", counter)
        counter += 1
        if counter >=10:
            break

    ffmpeg_cmd = "/usr/bin/ffmpeg -y -framerate 5 -pattern_type glob -i '{}/*.png' -c:v libx264  {}".format(output_dir, output_file)
    os.system(ffmpeg_cmd)
        
    return output_file

def run_bg_change_cv2(input_video, bg_image, text_prompt, box_threshold, text_threshold, filter_size):
    #base_dir = os.path.join("/tmp", str(uuid.uuid4()))
    base_dir = os.path.dirname(input_video)
    output_video_name = os.path.join(base_dir, 'output.mp4')
    capture = cv2.VideoCapture(input_video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second=capture.get(cv2.CAP_PROP_FPS)
    if frames_per_second is not None:
        save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), frames_per_second, (width, height))
    
    counter = 0
    start = time.time() 
    
    while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
            if counter % 5 != 0:
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            in_image = Image.fromarray(image)
            output, _ = run_grounded_sam(in_image, bg_image, text_prompt, box_threshold, text_threshold, filter_size)
            output = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
            print("No. of frames:", counter)
            if output_video_name is not None:
                save_video.write(output)
            if counter >= 50:
                break
        else:
          break

    capture.release()

    end = time.time()
    print(f"Processed {counter} frames in {end-start:.1f} seconds")
      
    if frames_per_second is not None:
        save_video.release()
        
    return output_video_name 
    
    
def run_grounded_sam(in_image, bg_image, text_prompt, box_threshold, text_threshold, filter_size):
    global blip_processor, blip_model, groundingdino_model, sam_predictor

    # load image
    image_pil = in_image.convert("RGB")
    bgimg_pil = bg_image.convert("RGB")
    transformed_image = transform_image(image_pil)

    if groundingdino_model is None:
        groundingdino_model = load_model(config_file, ckpt_filenmae, device=device)

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        groundingdino_model, transformed_image, text_prompt, box_threshold, text_threshold
    )

    size = image_pil.size
    bgimg_pil = bgimg_pil.resize(size)
    # process boxes
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]) # scale [0, 1) to [0, W) or [0, H)
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2 # change from (c_x, c_y) to (left, top), guess ?
        boxes_filt[i][2:] += boxes_filt[i][:2] # change from (w, h) to (right, bottom)

    boxes_filt = boxes_filt.cpu()

    if sam_predictor is None:
        # initialize SAM
        assert sam_checkpoint, 'sam_checkpoint is not found!'
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)

    image = np.array(image_pil)
    sam_predictor.set_image(image)

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = sam_predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        mask_input = None,
        multimask_output = False,
    )
    # masks: [1, 1, 512, 512]
    # merge
    masks = torch.sum(masks, dim=0).unsqueeze(0)
    masks = torch.where(masks > 0, True, False)
    mask = masks[0][0].cpu().numpy() # simply choose the first mask, which will be refine in the future release
    mask_pil = Image.fromarray(mask)
    if filter_size < 0:
        mask_pil = mask_pil.filter(ImageFilter.MinFilter(size=-filter_size)) # erosion
    if filter_size > 0:
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=filter_size)) # dilation
    
    mask_pil = ImageOps.invert(mask_pil) 
    image_pil.paste(bgimg_pil, (0, 0), mask_pil)
    return [image_pil, mask_pil]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    parser.add_argument('--port', type=int, default=30000, help='port to run the server')
    parser.add_argument('--no-gradio-queue', action="store_true", help='path to the SAM checkpoint')
    args = parser.parse_args()

    print(args)
    #debug
    #run_bg_change_ffmpeg("assets/1.mp4", Image.open("assets/bg.png"), "person", 0.3, 0.25, 5)

    block = gr.Blocks()
    if not args.no_gradio_queue:
        block = block.queue()

    with block:
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(source='upload', format='mp4', value="assets/1.mp4")
                bg_image = gr.Image(source='upload', type="pil", value="assets/bg.png")
                text_prompt = gr.Textbox(label="Text Prompt")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    filter_size = gr.Slider(
                        label="Erosion/Dilation Filter Size", minimum=-50, maximum=50, value=0, step=1
                    )

            with gr.Column():
                output_video = gr.Video()

        run_button.click(fn=run_bg_change_ffmpeg, inputs=[
                        input_video, bg_image, text_prompt, box_threshold, text_threshold, filter_size], outputs=output_video)


    block.launch(server_name='0.0.0.0', server_port=args.port, debug=args.debug, share=args.share, file_directories=['/tmp/'])