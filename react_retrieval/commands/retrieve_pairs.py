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


def collect_retrieved_images(file_list_controller, knns, output_dir, pool=None):
    tasks = []
    for idx in range(knns.shape[0]):
        subdir = os.path.join(output_dir, f'{idx:03d}')
        os.makedirs(subdir, exist_ok=True)

        for knn_idx in knns[idx]:
            image_file, text_file = file_list_controller[knn_idx]
            if pool is not None:
                tasks.append((image_file, text_file, subdir))
            else:
                copyfile((image_file, text_file, subdir))

    if pool is not None:
        pool.map(copyfile, tasks)

def main():
    args = get_argument_parser().parse_args()

    setup_logging(logging.INFO)

    with Timeit(f"Loading index: {args.faiss_index}"):
        index = faiss.read_index(args.faiss_index)

    with Timeit(f"Loading file list controller: {args.metafile}"):
        file_list_controller = FileListController(args.metafile, args.scatter_dir)

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

    pool = Pool(128)

    pbar = tqdm(class_names)
    for class_name in pbar:
        pbar.set_description(f'Fetching for {class_name}')
        texts = [template.format(class_name) for template in templates]

        dists, knns = query_by_text(texts, model, index, k=args.images_per_class)
        collect_retrieved_images(file_list_controller, knns,
            output_dir=os.path.join(args.output_dir, args.dataset, f'{args.images_per_class}nn', class_name), pool=pool)


if __name__ == "__main__":
    main()
