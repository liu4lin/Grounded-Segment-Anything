import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
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
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)
    
def copy_and_replace(orig_img, repl_img, mask_img):
     # Get the pixels of the main image, replacement image, and mask
    main_pixels = orig_img.load()
    repl_pixels = repl_img.load()
    mask_pixels = mask_img.load()
    # Replace the region in the main image with the replacement image, using the mask
    for x in range(orig_img.width):
        for y in range(orig_img.height):
            if mask_pixels[x, y]:
                main_pixels[x, y] = repl_pixels[x, y]

def image_crop_masked(orig_img, mask_img, offset=0.1):
    # Find the bounding box of the mask image
    y, x = np.where(np.array(mask_img) > 0)
    left, top = np.min(x), np.min(y)
    right, bottom = np.max(x), np.max(y)
    
    # Adjust the x,y coordinates dynamically
    offset_x = round((right - left) * offset * 0.5)
    offset_y = round((bottom - top) * offset * 0.5)
    left, right = left - offset_x, right + offset_x
    top, bottom = top - offset_y, bottom + offset_y

    # Crop the rectangle area of the original image based on the bounding box
    cropped_img = orig_img.crop((left, top, right, bottom))
    cropped_mask = mask_img.crop((left, top, right, bottom))
    return cropped_img, cropped_mask, left, top

def image_resize_bounded(input_image, max_side=None):
    if not isinstance(input_image, Image.Image):
        raise TypeError("input_image must be a PIL Image instance")

    # Determine the new size based on the maximum side
    width, height = input_image.size
    max_side = min(max_side, max(width, height))
    if width > height:
        new_width = max_side
        new_height = round(max_side * height / width)
    else:
        new_height = max_side
        new_width = round(max_side * width / height)

    # Ensure that the new height and width are multiples of 8
    new_height -= new_height % 8
    new_width -= new_width % 8

    # Resize the image
    new_size = (new_width, new_height)
    resized_image = input_image.resize(new_size)

    # Return the resized image
    return resized_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--det_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--inpaint_prompt", type=str, required=True, help="inpaint prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="save your huggingface large model cache")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--inpaint_mode", type=str, default="first", help="inpaint mode")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    det_prompt = args.det_prompt
    inpaint_prompt = args.inpaint_prompt
    output_dir = args.output_dir
    cache_dir=args.cache_dir
    # if not os.path.exists(cache_dir):
    #     print(f"create your cache dir:{cache_dir}")
    #     os.mkdir(cache_dir)
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    inpaint_mode = args.inpaint_mode
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, det_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # masks: [1, 1, 512, 512]

    # inpainting pipeline
    if inpaint_mode == 'merge':
        masks = torch.sum(masks, dim=0).unsqueeze(0)
        masks = torch.where(masks > 0, True, False)
    mask = masks[0][0].cpu().numpy() # simply choose the first mask, which will be refine in the future release
    mask_pil = Image.fromarray(mask) # ^ True) # inverse the detection
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=5)) # dilation for the objects
    image_pil = Image.fromarray(image)
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "/checkpoints/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # prompt = "A sofa, high quality, detailed"
    image_crop, mask_crop, left, top = image_crop_masked(image_pil, mask_pil, offset=0.1)
    image_inp = image_resize_bounded(image_crop, max_side=800)
    mask_inp = image_resize_bounded(mask_crop, max_side=800)
    width, height = image_inp.size
    #image_inp = image_pil.resize((720, 512))
    #mask_inp = mask_pil.resize((720, 512))
    image = pipe(prompt=inpaint_prompt, image=image_inp, mask_image=mask_inp, height=height, width=width).images[0]
    #image = pipe(prompt=inpaint_prompt, image=image_inp.resize((512, 512)), mask_image=mask_inp.resize((512, 512))).images[0]
    image = image.resize(image_crop.size)
    image.save(os.path.join(output_dir, "inpainting.jpg"))
    image_pil.paste(image, (left, top), mask_crop)
    image_pil.save(os.path.join(output_dir, "grounded_sam_inpainting_output.jpg"))
    mask_pil.save(os.path.join(output_dir, "mask.jpg"))

    # draw output image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # for box, label in zip(boxes_filt, pred_phrases):
    #     show_box(box.numpy(), plt.gca(), label)
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight")

