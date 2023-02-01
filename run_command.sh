# ===== stf training ====== #

# 2048 stf
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name 2048_stf_1_align_0 \
  --data=datasets/cifar10-32x32.zip --cond=0 --arch=ncsnpp \
  --pfgmv2=1 --batch 512 --align=0 --aug_dim 2048 --stf=1 --rbatch 1024

# 2048 condition
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name 2048_cond_1 \
  --data=datasets/cifar10-32x32.zip --cond=1 --arch=ncsnpp \
  --pfgmv2=1 --batch 512 --align=0 --aug_dim 2048



# ===== training =======
# 512
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name 512_stf_0_align_0 \
  --data=datasets/cifar10-32x32.zip --cond=0 --arch=ncsnpp \
  --pfgmv2=1 --batch 512 --align=0 --aug_dim 512
# 8192
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name 8192_stf_0_align_0 \
  --data=datasets/cifar10-32x32.zip --cond=0 --arch=ncsnpp \
  --pfgmv2=1 --batch 512 --align=0 --aug_dim 8192
# 3072000
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name 3072000_stf_0_align_0 \
  --data=datasets/cifar10-32x32.zip --cond=0 --arch=ncsnpp \
  --pfgmv2=1 --batch 512 --align=0 --aug_dim 3072000
# edm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name edm_ncsnpp \
  --data=datasets/cifar10-32x32.zip --cond=0 --arch=ncsnpp \
  --batch 512

# ===== generate =======
# 512
torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/512_stf_0_align_0 \
  --ckpt 100000 --pfgmv2=1 --align=0 --aug_dim=512
# 8192
torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/8192_stf_0_align_0 \
  --ckpt 100000 --pfgmv2=1 --align=0 --aug_dim=8192
# 3072000
torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/3072000_stf_0_align_0 \
  --ckpt 100000 --pfgmv2=1 --align=0 --aug_dim=3072000
# edm
torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/edm_ncsnpp \
  --ckpt 100000


# ===== fid evaluation =====
# 512
torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/512_stf_0_align_0 --ref=fid-refs/cifar10-32x32.npz --num 50000 --ckpt 100000 --gen_seed 1 > res_512_8GPU.txt
# 8192
torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/8192_stf_0_align_0 --ref=fid-refs/cifar10-32x32.npz --num 50000 --ckpt 100000 --gen_seed 1 > res_8192_8GPU.txt
# 3072000
torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/3072000_stf_0_align_0 --ref=fid-refs/cifar10-32x32.npz --num 50000 --ckpt 100000 --gen_seed 1 > res_3072000_8GPU.txt
# edm
torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/edm_ncsnpp --ref=fid-refs/cifar10-32x32.npz --num 50000 --ckpt 100000 --gen_seed 1 > res_edm.txt


# ================ Train FFHQ ================
# 128
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name ffhq_128_stf_0_align_0 \
  --data=datasets/ffhq-64x64.zip --cond=0 --arch=ddpmpp \
  --pfgmv2=1 --batch 256 --align=0 --aug_dim 128 \
  --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15
# generate
torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/ffhq_128_stf_0_align_0 \
  --ckpt 150000 --pfgmv2=1 --align=0 --aug_dim=128 --steps=40
# fid evaluation
torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/ffhq_128_stf_0_align_0 --ref=fid-refs/ffhq-64x64.npz --num 50000 --ckpt 150000 --gen_seed 1 > res_ffhq_ddpmpp_128.txt
