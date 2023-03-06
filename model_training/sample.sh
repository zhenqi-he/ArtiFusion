export OPENAI_LOGDIR="/root/autodl-tmp/models_diffusion"

MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 1"



python scripts/image_sample.py --model_path /root/autodl-tmp/models_diffusion/ema_0.9999_080000.pt $MODEL_FLAGS $DIFFUSION_FLAGS --num_sample=1