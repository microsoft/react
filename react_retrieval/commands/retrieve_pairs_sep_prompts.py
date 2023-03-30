import argparse
from tqdm import tqdm

import faiss
import torch
import torch.nn.functional as F
import clip
import numpy as np
import os
import shutil
from functools import partial

from react_retrieval.utils.prompts import *
from react_retrieval.data.filelist_controller import FileListController

from autofaiss.utils.decorators import Timeit

from multiprocessing.pool import ThreadPool as Pool

import logging

import re
import json


def setup_logging(logging_level: int):
    """Setup the logging."""
    logging.config.dictConfig(dict(version=1, disable_existing_loggers=False))
    logging_format = "%(asctime)s [%(levelname)s]: %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)


def get_argument_parser():
    parser = argparse.ArgumentParser(description='Extract features from Vision-Language datasets.')
    parser.add_argument('--model', type=str, default='vitb32')
    parser.add_argument('--metafile', type=str)
    parser.add_argument('--scatter_dir', type=str)
    parser.add_argument('--faiss_index', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset', type=str, help='ICinW dataset name.')
    parser.add_argument('--images_per_class', type=int, default=50, help='number of images per class')
    return parser


def query_by_text(texts, model, index, k):
    assert type(texts) is list, "Please wrap texts into List[str]."

    texts = clip.tokenize(texts, context_length=77, truncate=True).cuda()
    with torch.no_grad():
        text_embeddings = model.encode_text(texts)
        text_embeddings = F.normalize(text_embeddings)

    queries = text_embeddings.data.cpu().numpy().astype('float32')
    dists, knns = index.search(queries, k)
    
    return dists, knns


def copyfile(args):
    image_file, text_file, subdir = args
    shutil.copy2(image_file, os.path.join(subdir, os.path.basename(image_file)))
    shutil.copy2(text_file, os.path.join(subdir, os.path.basename(text_file)))


def format_cls_name(cls_name):
    new_name = cls_name
    new_name = re.sub(r'[^0-9a-zA-Z_]', '_', new_name)
    new_name = re.sub(r'_+', '_', new_name).strip('_')
    return new_name


def main():
    args = get_argument_parser().parse_args()

    setup_logging(logging.INFO)

    with Timeit(f"Loading index: {args.faiss_index}"):
        index = faiss.read_index(args.faiss_index)

    with Timeit(f"Loading file list controller: {args.metafile}"):
        file_list_controller = FileListController(args.metafile, args.scatter_dir, is_tar=True)

    class_names = class_map[args.dataset]
    templates = template_map[args.dataset]

    model_name = {
        "vitb32": "ViT-B/32",
        "vitb16": "ViT-B/16",
        "vitl14": "ViT-L/14",
    }[args.model]

    with Timeit(f"Loading model: {model_name}"):
        if model_name in clip.available_models():
            model, preprocess = clip.load(model_name)
        model.cuda().eval()

    KNNS, DISTS, LABELS = [], [], []
    for class_idx, class_name in enumerate(class_names):
        texts = [template.format(class_name) for template in templates]
        dists, knns = query_by_text(texts, model, index, k=args.images_per_class)
        KNNS.append(knns)
        DISTS.append(dists)
        LABELS.append(np.full_like(knns, class_idx, dtype=np.int64))

    KNNS = np.stack(KNNS)
    DISTS = np.stack(DISTS)
    LABELS = np.stack(LABELS)

    n_classes, n_prompts, n_neighbors = KNNS.shape

    output_image_dir = os.path.join(args.output_dir, 'images', args.dataset)
    output_meta_dir = os.path.join(args.output_dir, 'metas', args.dataset)
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_meta_dir, exist_ok=True)

    TASKS = []

    for cls_idx in range(n_classes):
        class_name = class_names[cls_idx]
        if type(class_name) == list:
            class_name = '_and_'.join(class_name)
        class_name = format_cls_name(class_name)
        cls_images = os.path.join(output_image_dir, class_name)
        os.makedirs(cls_images, exist_ok=True)

        metadata = dict()

        for prompt_idx in range(n_prompts):
            for neighbor_idx in range(n_neighbors):
                index = KNNS[cls_idx, prompt_idx, neighbor_idx]
                dist = DISTS[cls_idx, prompt_idx, neighbor_idx].tolist()
                image_file, text_file = file_list_controller[index]
                filename = os.path.basename(image_file)
                if filename not in metadata:
                    metadata[filename] = {'search_meta': []}
                    TASKS.append((image_file, text_file, cls_images))
                cur_meta = metadata[filename]
                query_class = class_names[cls_idx]
                if type(query_class) == list:
                    query_class = query_class[0]
                cur_meta['search_meta'].append({
                    'query': templates[prompt_idx].format(query_class),
                    'dist': dist,
                })

        with open(os.path.join(output_meta_dir, f'{class_name}.json'), 'w') as fp:
            json.dump(metadata, fp)

    with Pool(128) as pool:
        r = list(tqdm(pool.imap(copyfile, TASKS), total=len(TASKS)))

if __name__ == "__main__":
    main()
