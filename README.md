# PFGM++: Unlocking the Potential of Physics-Inspired Generative Models

Pytorch implementation of the paper PFGM++: Unlocking the Potential of Physics-Inspired Generative Models

by [Yilun Xu](http://yilun-xu.com), [Ziming Liu](https://kindxiaoming.github.io/#pub), [Yonglong Tian](https://people.csail.mit.edu/yonglong/), Shangyuan Tong [Max Tegmark](https://space.mit.edu/home/tegmark/), [Tommi S. Jaakkola](http://people.csail.mit.edu/tommi/)



😇 *Improvement over PFGM / Diffusion Models*

- No longer require the large batch training target in *PFGM*
- More general $D \in \mathbb{R}^+$ dimensional augmented variable. PFGM++ subsumes PFGM and Diffusion Models: *PFGM* correspond to $D=1$ and *Diffusion Models* correspond to $D\to \infty$.
- Existence of sweet spot $D^*$ in the middle!
- Smaller $D$ more robust than *Diffusion Models* ( $D\to \infty$ )
- Enable the adjustment for model robustness and rigidity!

---



*Abstract:* We present a general framework termed *PFGM++* that unifies diffusion models and Poisson Flow Generative Models (PFGM). These models realize generative trajectories for $N$ dimensional data by embedding paths in $N{+}D$ dimensional space while still controlling the progression with a simple scalar norm of the $D$ additional variables. The new models reduce to **PFGM when $D{=}1$** and to **diffusion models when $D{\to}\infty$.** The flexibility of choosing $D$ allows us to trade off robustness against rigidity as increasing $D$ results in more concentrated coupling between the data and the additional variable norms. We **dispense with the biased large batch field targets used in PFGM and instead provide an unbiased perturbation-based objective** similar to diffusion models. To explore different choices of $D$, we provide a direct alignment method for transferring well-tuned hyperparameters from diffusion models ( $D{\to} \infty$ ) to any finite $D$ values. Our experiments show that models with **finite $D$ can be superior to previous state-of-the-art diffusion models** on CIFAR-10/FFHQ $64{\times}64$ datasets, with FID scores of $1.91/2.43$ when $D{=}2048/128$. In addition, we demonstrate that models with smaller $D$ exhibit **improved robustness** against modeling errors.

![schematic](assets/pfgmpp.png)

---



## Outline

Our implementation is built upon the [EDM](https://github.com/NVlabs/edm) repo. We first provide an [guidance](#quick-adoptation) on how to quickly **transfer the hyperparameter from well-tuned diffusion models** ( $D\to \infty$ ), such as **EDM** and **DDPM**, **to the PFGM++ family** ( $D\in \mathbb{R}^+$ ) in a task/dataset agnostic way (We provide more details in *Sec 4 ( Transfer hyperparameters to finite $D$s ) and Appendix C.2* in our paper). We highlight our modifications based on their original command lines for [training](#training-new-models-with-stf), [sampling and evaluation](#generate-&-evaluations). 

We also provide the original instruction for [set-ups](#the-instructions-for-set-ups-from-edm-repo), such as environmental requirements and dataset preparation, from EDM repo.



## Transfer guidance by $r=\sigma\sqrt{D}$ formula

Below we provide the guidance for how to quick transfer the well-tuned hyperparameters for diffusion models ( $D\to \infty$ ), such as  $\sigma_{\textrm{max}}$ and $p(\sigma)$ to finite $D$s. We adopt the $r=\sigma\sqrt{D}$ formula in our paper for the alignment (c.f. Section 4).

Training hyperparameter transfer. The example we used is a simplified version of  [`loss.py`]([https://github.com/Newbeeer/stf/blob/13de0c799a37dd2f83108c1d7295aaf1e993dffe/training/loss.py#L78-L118) in this repo.

```python

```

Sampling hyperparameter transfer. The example we used is a simplified version of  [`generate.py`]([https://github.com/Newbeeer/stf/blob/13de0c799a37dd2f83108c1d7295aaf1e993dffe/training/loss.py#L78-L118) in this repo.

```python

```

Please refer to **Appendix C.2** for detailed hyperparameter transfer procedures from **EDM** and **DDPM​**.



## Training PFGM++

You can train new models using `train.py`. For example:

```sh
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs --name exp_name \
--data=datasets/cifar10-32x32.zip --cond=0 --arch=arch \
--pfgmpp=1 --batch 512 \
--aug_dim aug_dim

exp_name: name of experiments
aug_dim: D (additional dimensions)  
arch: model architectures. options: ncsnpp | ddpmpp
pfgmpp: use PFGM++ framework, otherwise diffusion models (D\to\infty case). options: 0 | 1
```

The above example uses the default batch size of 512 images (controlled by `--batch`) that is divided evenly among 8 GPUs (controlled by `--nproc_per_node`) to yield 64 images per GPU. Training large models may run out of GPU memory; the best way to avoid this is to limit the per-GPU batch size, e.g., `--batch-gpu=32`. This employs gradient accumulation to yield the same results as using full per-GPU batches. See [`python train.py --help`](./docs/train-help.txt) for the full list of options.

The results of each training run are saved to a newly created directory  `training-runs/exp_name` . The training loop exports network snapshots `training-state-*.pt`) at regular intervals (controlled by  `--dump`). The network snapshots can be used to generate images with `generate.py`, and the training states can be used to resume the training later on (`--resume`). Other useful information is recorded in `log.txt` and `stats.jsonl`. To monitor training convergence, we recommend looking at the training loss (`"Loss/loss"` in `stats.jsonl`) as well as periodically evaluating FID for `training-state-*.pt` using `generate.py` and `fid.py`.

For FFHQ dataset, replacing `--data=datasets/cifar10-32x32.zip` with `--data=datasets/ffhq-64x64.zip`

**Sidenote:** The original EDM repo provide more dataset: FFHQ, AFHQv2, ImageNet-64. We did not test the performance of *PFGM++* on these datasets due to limited computational resources. However, we believe that the **some finte $D$s (sweet spots) would beat the diffusion models (the $D\to\infty$ case)**. Please let us know if you have those resutls 😀



TODO: All checkpoints are provided in this [Google drive folder](https://drive.google.com/drive/folders/1bTtRCkl31VP6KC71l5kvXCLE4NT5kVtu?usp=share_link).



## Generate & Evaluations

- Generate 50k samples:

  ```zsh
  torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/exp_name \
  --pfgmpp=1 --aug_dim=aug_dim
     
  exp_name: name of experiments
  aug_dim: D (additional dimensions)  
  arch: model architectures. options: ncsnpp | ddpmpp
  pfgmpp: use PFGM++ framework, otherwise diffusion models (D\to\infty case). options: 0 | 1
  ```
  
Note that the numerical value of FID varies across different random seeds and is highly sensitive to the number of images. By default, `fid.py` will always use 50,000 generated images; providing fewer images will result in an error, whereas providing more will use a random subset. To reduce the effect of random variation, we recommend repeating the calculation multiple times with different seeds, e.g., `--seeds=0-49999`, `--seeds=50000-99999`, and `--seeds=100000-149999`. In the EDM paper, they calculated each FID three times and reported the minimum.
  
For the FID versus controlled $\alpha$/NFE/quantization, please use `generate_alpha.py/generate_steps.py/generate_quant.py` for generation.
  
- FID evaluation

  ```zsh
  torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/exp_name --ref=fid-refs/cifar10-32x32.npz --num 50000 
  
  exp_name: name of experiments
  ```

  



## The instructions for set-ups from EDM repo

### Requirements

- Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
- 1+ high-end NVIDIA GPU for sampling and 8+ GPUs for training. We have done all testing and development using V100 and A100 GPUs.

- 64-bit Python 3.8 and PyTorch 1.12.0 (or later). See [https://pytorch.org](https://pytorch.org/) for PyTorch install instructions.
- Python libraries: See `environment.yml`for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda env create -f environment.yml -n edm`
  - `conda activate edm`
- Docker users:
  - Ensure you have correctly installed the [NVIDIA container runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu).
  - Use the [provided Dockerfile](https://github.com/NVlabs/edm/blob/main/Dockerfile) to build an image with the required library dependencies.

### Preparing datasets

Datasets are stored in the same format as in [StyleGAN](https://github.com/NVlabs/stylegan3): uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels. Custom datasets can be created from a folder containing images; see [`python dataset_tool.py --help`](./docs/dataset-tool-help.txt) for more information.

**CIFAR-10:** Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```

**FFHQ:** Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as 1024x1024 images and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=downloads/ffhq/images1024x1024 \
    --dest=datasets/ffhq-64x64.zip --resolution=64x64
python fid.py ref --data=datasets/ffhq-64x64.zip --dest=fid-refs/ffhq-64x64.npz
```

**AFHQv2:** Download the updated [Animal Faces-HQ dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) (`afhq-v2-dataset`) and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=downloads/afhqv2 \
    --dest=datasets/afhqv2-64x64.zip --resolution=64x64
python fid.py ref --data=datasets/afhqv2-64x64.zip --dest=fid-refs/afhqv2-64x64.npz
```

**ImageNet:** Download the [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \
    --dest=datasets/imagenet-64x64.zip --resolution=64x64 --transform=center-crop
python fid.py ref --data=datasets/imagenet-64x64.zip --dest=fid-refs/imagenet-64x64.npz
```

