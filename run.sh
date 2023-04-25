export HUGGINGFACE_HUB_CACHE=/checkpoints/hf

sudo docker run --gpus all -it --rm --net=host --privileged \
--env HUGGINGFACE_HUB_CACHE="/checkpoints/hf" \
-v $PWD:/home/appuser -v /data-cbs1/ubuntu/checkpoints:/checkpoints \
-v $PWD/tmp:/tmp -w /home/appuser gsam:v1 bash

python -c 'import gradio; gradio.close_all()'

python gradio_app.py --port 30000

sudo docker run --gpus all -it --rm --net=host --privileged \
--env HUGGINGFACE_HUB_CACHE="/checkpoints/hf" \
-v $PWD:/home/appuser -v /data-cbs1/ubuntu/checkpoints:/checkpoints \
-w /home/appuser gsam:v1 \
python grounded_sam_inpainting_fg.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /checkpoints/GroundingDINO/groundingdino_swint_ogc.pth \
  --sam_checkpoint /checkpoints/SAM/sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --det_prompt "bears, animals" \
  --inpaint_prompt "dogs, high quality, detailed" \
  --inpaint_mode "merge" \
  --device "cuda"






  python grounded_sam_inpainting_bg.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /checkpoints/GroundingDINO/groundingdino_swint_ogc.pth \
  --sam_checkpoint /checkpoints/SAM/sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --det_prompt "bears, animals" \
  --inpaint_prompt "wetland, high quality, detailed" \
  --inpaint_mode "merge" \
  --device "cuda"