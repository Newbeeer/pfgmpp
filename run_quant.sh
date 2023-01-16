# ===== generate =======
# 128
CUDA_VISIBLE_DEVICES=0,2,6,7 torchrun --standalone --nproc_per_node=4 generate_quant.py \
  --seeds=0-49999 --outdir /scratch/ylxu/edm/best_128 \
  --pfgmv2=1 --align=0 --aug_dim=128


CUDA_VISIBLE_DEVICES=0,2,6,7 torchrun --standalone --nproc_per_node=4 generate_quant.py \
  --seeds=0-49999 --outdir /scratch/ylxu/edm/best_2048 \
  --pfgmv2=1 --align=0 --aug_dim=2048


CUDA_VISIBLE_DEVICES=0,2,6,7 torchrun --standalone --nproc_per_node=4 generate_quant.py \
  --seeds=0-49999 --outdir /scratch/ylxu/edm/best_edm_ve \
  --edm=1


