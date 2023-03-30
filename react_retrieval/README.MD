# :globe_with_meridians: REACT: Retrieval Stage


- [1. Download Retrieved Pairs for Standard Benchmarks](#1.-Download-Retrieved-Pairs-for-Standard-Benchmarks)
- [2. Build and Retrieve on Customized Image-Text Pairs](#2.-Build-and-Retrieve-from-Customized-Image-Text-Pairs)
  - [2.1 Download CC3M Dataset](#2.1-Download-CC3M-Dataset) 
  - [2.2 Build Index](#2.2-Build-Index)
  - [2.3 Retrieve Image-Text Pairs](#2.3-Retrieve-Image-Text-Pairs)

## Installation

```Shell
conda create -n react_retrieval python=3.7 -y
conda activate react_retrieval
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e .
```

## 1. Download Retrieved Pairs for Standard Benchmarks

If you want to focus on building customized models on standard benchmarks like ImageNet-1K and ELEVATER, you may skip the following steps and directly download our retrieved pairs.

See instructions [here](../react_customization).

## 2. Build and Retrieve from Customized Image-Text Pairs
We use CC3M as an example to demonstrate how to build indexing system and retrieve from the indexed dataset.

### 2.1 Download CC3M Dataset

Follow the instructions [here](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md) to download CC3M dataset.  You can also find [examples](https://github.com/rom1504/img2dataset#examples) for other datasets.

### 2.2 Build Index
We use CC-3M as an example for the retrieval system itself, as it is much smaller and more accessible for all kinds of hardware setups.

1. Extract features.  This step may take a while.  You can split the dataset into multiple chunks by specifying `--tsv_chunks` and run the script in parallel to speed up the process.

```Shell
python commands/extract_feature_pairs.py \
    --model 'ViT-B/32' \
    --dataset_dir '/path/to/CC3M' \
    --save_dir '/path/to/save/features' \
    --tsv_chunk_idx=0 \
    --tsv_chunks=1 \
    --tsv_padzero=5 \
    --batch_size=128 \
    --workers=8
```

2. Train index.

Note that `--feature_mode` can be either `image` or `text`.  If you want to build a retrieval system with texts as the **keys**, you can set `--feature_mode=text`.  For example, for T2I retrieval, we use `feature_mode=image`.

```Shell
python commands/train_index.py \
    --d=512 \
    --dataset_size=3000000 \
    --metafile '/path/to/save/features/metafile.json' \
    --data_root '/path/to/save/features' \
    --index_train_features '/path/to/save/features/train_features.npy' \
    --faiss_index '/path/to/save/index' \
    --base_index_path '/path/to/save/base_index' \
    --feature_mode '{image,text}'
```

### 2.3 Retrieve Image-Text Pairs

After the index is built, you can retrieve image-text pairs efficiently.

#### Preprocess

1. Create filelist for fast indexing between multiple chunked files.

```Shell
python create_filelist_webdataset.py \
    --dataset_dir '/path/to/CC3M' \
    --filelist '/path/to/save/filelist.pkl'
```

2. We find scattering the retrieved pairs (originally in `tar` format) to separate files typically gives us better throughput.

```Shell
python extract_webdataset_scattered.py \
    --dataset_dir '/path/to/CC3M' \
    --scatter_dir '/path/to/save/scattered' \
    --tsv_padzero=5
```

#### Examples
We provide two examples for retrieval:
1. a Jupyter notebook ([here](./notebook/retrieve_pairs.ipynb)) for exploring the retrieval results interactively, and
2. a sample script (see below) that allows you to retrieve image-text pairs in batch.

```Shell
python commands/retrieve_pairs_sep_prompts.py \
    --metafile '/path/to/save/filelist.pkl' \
    --scatter_dir '/path/to/save/scattered' \
    --faiss_index '/path/to/save/index' \
    --output_dir '/path/to/save/retrieved_pairs' \
    --dataset caltech-101 \
    --images_per_class 200
```
