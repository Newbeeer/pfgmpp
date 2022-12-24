CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp \
--seeds=0-63 --subdirs  --network=training-runs/cifar_stf_1024_2/training-state-022579.pt


CUDA_VISIBLE_DEVICES=7  torchrun --standalone --nproc_per_node=1 fid.py calc --images=./training-runs/cifar_stf_1024_2/ckpt_22579 \
    --ref=../data/cifar10/cifar10-32x32.npz