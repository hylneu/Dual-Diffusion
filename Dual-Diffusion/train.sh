export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="/home/hyl/hou/DisenBooth-main/dataset/dog2/"
export OUTPUT_DIR="./output-dog2_newmodel_snow"
accelerate launch train.py\
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a dog</w> dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=200 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --validation_prompt="a dog </w> dog in the snow." \
  --validation_epochs=200 \
  --seed="0" \

#accelerate launch train_cl.py\

