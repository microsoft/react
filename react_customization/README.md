# :art: REACT: Customization Stage

**Disclaimer**: *The initial development of REACT customization originated from an internal code base at Microsoft. For public release, we adapted our work to the open-source OpenCLIP code base. Although there might be subtle differences in implementation and training, we have confirmed that the final results are comparable when using the CLIP ViT-B/32 backbone (68.4 this implementation vs 68.6 in paper).  See [below](#details-in-the-adaptation-to-openclip-codebase) for details.*

- [Evaluation](#evaluation)
  - [Evaluating ImageNet-1K on OpenCLIP](#evaluating-imagenet-1k-on-openclip)
  - [Evaluating ImageNet-1K and ELEVATER benchmark using ELEVATER Toolkit](#evaluating-imagenet-1k-and-elevater-benchmark-using-elevater-toolkit)
- [Training](#training)
  - [Download Retrieved Pairs](#download-retrieved-pairs)
  - [Gated-image Locked-text Tuning](#gated-image-locked-text-tuning-on-8x-v100s)
  - [Locked-text Tuning](#locked-text-tuning-on-8x-v100s)
- [Details in the adaptation to OpenCLIP codebase](#details-in-the-adaptation-to-openclip-codebase)


## Installation
```
conda create -n react python=3.9 -y
conda activate react
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
make install-training
make install
pip install tensorboard future wandb
```

## Evaluation

We support evaluation of ImageNet-1K both *directly* using OpenCLIP and using the official toolkit of ELEVATER benchmark.  We have verified that there are no significant differences in performance when using OpenCLIP or ELEVATER for evaluation(<0.1% for almost all cases).

For paper's results, we used the official toolkit of ELEVATER benchmark for evaluating on both ImageNet-1K and ELEVATER 20 datasets.

### Evaluating ImageNet-1K on OpenCLIP

We support evaluation of ImageNet-1K *directly* using OpenCLIP.

Specifying the pretrained checkpoints with `--model` and `--pretrained` allows automatically downloading the pretrained checkpoints from Hugging Face Hub.

`--model` can be one of the following:
- Locked-Text Tuning: `ViT-B-32`, `ViT-B-16`, `ViT-L-14`, `ViT-G-14`
- Gated-Image Tuning: `react_ViT-B-32`, `react_ViT-B-16`, `react_ViT-L-14`, `react_ViT-G-14`

`--pretrained` follows the following format: `react_{base_ckpt}_ret_{retrieval_set}`:
- `base_ckpt` should be one of the following: `clip`, `openclip_laion400m`, and `openclip_laion2b`;
- `retrieval_set`should be one of the following: `laion400m`, `laion2b`.

See valid combinations from our [list of pretrained checkpoints](../#pretrained-models).

 An example of evaluating **CLIP ViT-B/32** checkpoint customized with REACT on **LAION-400M** using gated-image tuning is provided below.

```Shell
python -m training.main \
  --zeroshot-frequency 1 \
  --imagenet-val=/path/to/imagenet/val/ \
  --batch-size=256 \
  --workers=16 \
  --model react_ViT-B-32 \
  --pretrained react_clip_ret_laion400m
```

### Evaluating ImageNet-1K and ELEVATER benchmark using ELEVATER Toolkit

We used the official toolkit of ELEVATER benchmark for evaluating on both ImageNet-1K and ELEVATER 20 datasets.  Please refer to the official documentation of ELEVATER for [installation](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC#installation) and [evaluation](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC#evaluation).

We provide a sample script for zero-shot evaluation on ImageNet-1K with CLIP ViT-B/32 customized with REACT using gated-image tuning on LAION-400M.

```Shell
cd vision_benchmark

python commands/zeroshot.py \
  --ds resources/datasets/imagenet-1k.yaml \
  --model resources/model/react_vitb32_CLIP.yaml \
  --save-predictions False \
  MODEL.CLIP_FP32 False
```

## Training

### Download Retrieved Pairs

1. Download the retrieved pairs meta data from [here](https://huggingface.co/datasets/react-vl/react-retrieval-datasets/blob/main/imagenet_10m.parquet).  This is a parquet file with 10M retrieved pairs from LAION-400M dataset for ImageNet-1K.

2. Install [img2dataset](https://github.com/rom1504/img2dataset): `pip install img2dataset`

3. Set up DNS resolver following [img2dataset](https://github.com/rom1504/img2dataset#setting-up-a-high-performance-dns-resolver) guidelines.  It is crucial for a high success rate for retrieval so as to have a retrieved dataset that is as complete as possible!

4. Download the pairs dataset with `img2dataset`.

```Shell
img2dataset --url_list ./imagenet_10m.parquet --input_format "parquet"\
  --url_col "URL" --caption_col "TEXT" --output_format webdataset\
    --output_folder ./imagenet_10m --processes_count 64 --thread_count 12 --image_size 384 \
    --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
    --enable_wandb True
```

### Gated-image Locked-text tuning on 8x V100s
```Shell
torchrun --nproc_per_node 8 -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --name React_gi-clip_b32-lr_5e-4-b_512-grad_ckpt-v100_8-aug_on-p_amp-wds-wk_4 \
    --train-data '/path/to/imagenet_10m/{00000..00897}.tar' \
    --dataset-type webdataset \
    --train-num-samples 8530639 \
    --imagenet-val=/path/to/imagenet/val/ \
    --warmup 5000 \
    --batch-size=512 \
    --grad-checkpointing \
    --lr=5e-4 \
    --wd=0.1 \
    --epochs=32 \
    --workers=4 \
    --model react_ViT-B-32-quickgelu \
    --resume ./checkpoints/ViT-B-32.pt \
    --lock-image-unlock-gated \
    --lock-text \
    --aug-cfg \
      use_timm=True \
      scale='(0.08,1.0)' \
      ratio='(0.75,1.3333333)' \
      hflip=0.5 \
      interpolation=bicubic \
      color_jitter=0.4 \
      "auto_augment='rand-m9-mstd0.5-inc1'" \
      re_prob=0.25 \
      re_count=1
```


### Locked-text tuning on 8x V100s
```Shell
torchrun --nproc_per_node 8 -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --name React_lt-clip_b32-lr_5e-5-b_512-grad_ckpt-v100_8-aug_on-p_amp-wds-wk_4 \
    --train-data '/path/to/imagenet_10m/{00000..00897}.tar' \
    --dataset-type webdataset \
    --train-num-samples 8530639 \
    --imagenet-val=/path/to/imagenet/val/ \
    --warmup 5000 \
    --batch-size=512 \
    --grad-checkpointing \
    --lr=5e-5 \
    --wd=0.1 \
    --epochs=32 \
    --workers=4 \
    --model ViT-B-32-quickgelu \
    --resume ./checkpoints/ViT-B-32.pt \
    --lock-text \
    --aug-cfg \
      use_timm=True \
      scale='(0.08,1.0)' \
      ratio='(0.75,1.3333333)' \
      hflip=0.5 \
      interpolation=bicubic \
      color_jitter=0.4 \
      "auto_augment='rand-m9-mstd0.5-inc1'" \
      re_prob=0.25 \
      re_count=1
```

## Details in the adaptation to OpenCLIP codebase

The initial development of REACT customization originated from an internal code base at Microsoft. For public release, we adapted our work to the open-source OpenCLIP code base. Although there might be subtle differences in implementation and training, we have confirmed that the final results are comparable when using the CLIP ViT-B/32 backbone.  We provide more details here.

The primary cause for the difference in results is due to changes in data availability on the web over time. Our internal implementation utilized a retrieval set from the LAION-400M dataset in December 2021. Upon re-implementation with OpenCLIP in February 2023, we found that some retrieved pairs were no longer available online, resulting in a reduced "Feb 2023 set" containing only ~8.5M pairs, compared to the initial 10M in "Dec 2021 set".

We trained REACT using CLIP ViT-B/32 backbone on OpenCLIP with both "Dec 2021 set" and "Feb 2023 set", yielding comparable results. For "Dec 2021 set", there was a 0.2% performance drop with gated-image tuning and a 0.1% improvement with locked-text tuning. On "Feb 2023 set", despite utilizing only 85% of the original data, there was a minor 0.4% performance drop with gated-image tuning and equivalent performance with locked-text tuning. Detailed results are provided below, along with training logs on [wandb.ai](https://api.wandb.ai/links/lht/nu5eexpi).

As an effort to facilitate the research community, we are seaking to release the image-text pairs that we retrieved from LAION-400M dataset in Dec 2021.  We will update this repo once the data is released.

|             | Paper (Dec 2021 set) | OpenCLIP (Dec 2021 set) | OpenCLIP (Feb 2023 set) |
|-------------|----------------------|-------------------------|-------------------------|
| Baseline    | 63.2                 | 63.2                    | 63.2                    |
| Locked-text | 66.9                 | 67.0                    | 66.9                    |
| Gated-image | 68.6                 | 68.4                    | 68.2                    |


