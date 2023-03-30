import argparse
from tqdm import tqdm

import json
import os
import faiss
import time

from collections import OrderedDict

import numpy as np

from react_retrieval.data.datasets import TSVFile, TSVDataset, decode_pairs_feature
from react_retrieval.index import index_utils

from autofaiss.utils.decorators import Timeit

import logging

def setup_logging(logging_level: int):
    """Setup the logging."""
    logging.config.dictConfig(dict(version=1, disable_existing_loggers=False))
    logging_format = "%(asctime)s [%(levelname)s]: %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)

def get_argument_parser():
    parser = argparse.ArgumentParser(description='Extract features from Vision-Language datasets.')
    parser.add_argument('--d', default=512, type=int)
    parser.add_argument('--dataset_size', default=3_000_000, type=int)
    parser.add_argument('--metafile', type=str)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--index_train_features', type=str)
    parser.add_argument('--faiss_index', required=True, type=str)
    parser.add_argument('--base_index_path', type=str)
    parser.add_argument('--feature_mode', type=str)
    parser.add_argument('--index_key', default=None, type=str)
    parser.add_argument('--metric_type', default="IP", type=str)
    parser.add_argument('--select_n_per_tsv', default=16384, type=int)
    return parser


def sample_train_features(args):
    with open(args.metafile, "r") as fp:
        metafile = json.load(fp, object_pairs_hook=OrderedDict)

    n_total_entries = len(metafile) * args.select_n_per_tsv
    n_total_entries = min(n_total_entries, sum(metafile.values()))
    all_features = np.zeros((n_total_entries, args.d), dtype=np.float32)
    pbar = tqdm(sorted(metafile.keys()))
    current_idx = 0
    processed_tsv = 0
    os.makedirs(os.path.dirname(args.index_train_features), exist_ok=True)
    for tsv_idx in pbar:
        num_entries = metafile[tsv_idx]
        tsv_file_name = f'{tsv_idx}.tsv'
        tsv_file_path = os.path.join(args.data_root, tsv_file_name)
        pbar.set_description(f'Current: {tsv_file_name}, Loaded: {current_idx} vectors, Processed: {processed_tsv} files')
        dataset = TSVDataset(tsv_file_path, transform=decode_pairs_feature)
        select_n_per_tsv = min(args.select_n_per_tsv, num_entries)
        sample_idx = np.arange(num_entries)
        if num_entries > select_n_per_tsv:
            sample_idx = np.random.choice(sample_idx, size=select_n_per_tsv, replace=False)
        results = [dataset[idx] for idx in sample_idx]
        if args.feature_mode == 'image':
            features = [x[2].astype(np.float32) for x in results]
        elif args.feature_mode == 'text':
            features = [x[3].astype(np.float32) for x in results]
        else:
            assert False, f"Unknown feature type: {args.feature_mode}"
        features = np.stack(features, axis=0)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        all_features[current_idx:current_idx+select_n_per_tsv] = features
        current_idx += select_n_per_tsv
        processed_tsv += 1
    np.save(args.index_train_features, all_features)
    return all_features


def build_base_index(args):
    if args.index_key is None:
        args.index_key = index_utils.get_best_index(args.dataset_size, args.d)
    base_index = index_utils.create_empty_index(args.d, args.index_key, args.metric_type)
    if index_utils.check_if_index_needs_training(args.index_key):
        assert args.index_train_features is not None, "Need to train index, `--index_train_features` must not be empty"
        if os.path.isfile(args.index_train_features):
            print(f'Loading train features')
            start = time.time()
            train_features = np.load(args.index_train_features)
            print('Loaded train features in {} s'.format(time.time() - start))
            if train_features.dtype == np.float16:
                print('Find train features in np.float16, converting to np.float32')
                train_features = train_features.astype('float32')
                np.save(args.index_train_features, train_features)
        else:
            train_features = sample_train_features(args)

        print(f'Training index on {train_features.shape[0]} features')
        start = time.time()
        base_index.train(train_features)
        print('Training took {} s'.format(time.time() - start))

        print('Writing index after training')
        start = time.time()
        faiss.write_index(base_index, args.base_index_path)
        print('Writing index took {} s'.format(time.time()-start))

    return base_index


def retrieve_features(tsv_idx, args):
    tsv_file_name = f'{tsv_idx}.tsv'
    tsv_file_path = os.path.join(args.data_root, tsv_file_name)
    dataset = TSVDataset(tsv_file_path, transform=decode_pairs_feature)
    results = [dataset[idx] for idx in range(len(dataset))]
    if args.feature_mode == 'image':
        features = [x[2].astype(np.float32) for x in results]
    elif args.feature_mode == 'text':
        features = [x[3].astype(np.float32) for x in results]
    else:
        assert False, f"Unknown feature type: {args.feature_mode}"
    features = np.stack(features, axis=0)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features


def create_meta_file(args):
    tsv_list = sorted(x.replace('.tsv', '') for x in os.listdir(args.data_root) if x.endswith('.tsv'))

    metadata = OrderedDict()
    for tsv_idx in tqdm(tsv_list):
        metadata[tsv_idx] = len(TSVFile(os.path.join(args.data_root, f'{tsv_idx}.tsv')))

    with open(args.metafile, "w") as fp:
        json.dump(metadata, fp)


def add_features_to_index(trained_index, args):
    with open(args.metafile, "r") as fp:
        metafile = json.load(fp, object_pairs_hook=OrderedDict)

    tsv_indices = sorted(metafile.keys())

    pbar = tqdm(range(len(tsv_indices)))
    for idx in pbar:
        tsv_idx = tsv_indices[idx]
        pbar.set_description(f'{tsv_idx}.tsv')

        features = retrieve_features(tsv_idx, args)
        trained_index.add(features)

    with Timeit(f"Saving index to disk", indent=0):
        faiss.write_index(trained_index, args.faiss_index)
    return trained_index


def main():
    args = get_argument_parser().parse_args()

    setup_logging(logging.INFO)

    if not os.path.isfile(args.metafile):
        create_meta_file(args)

    if os.path.isfile(args.base_index_path):
        base_index = faiss.read_index(args.base_index_path)
    else:
        base_index = build_base_index(args)

    add_features_to_index(base_index, args)

if __name__ == "__main__":
    main()
