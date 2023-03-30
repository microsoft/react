import os
import pickle
from collections import OrderedDict
import numpy as np


class FileListController:
    def __init__(self, metafile, scatter_dir, is_tar=False) -> None:
        self.scatter_dir = scatter_dir
        self.is_tar = is_tar

        with open(metafile, "rb") as fp:
            self.meta = pickle.load(fp)
            self.cumsum = self.get_cumsum()

    def __getitem__(self, index):
        tsv_idx, sample_idx = self.parse_index(index)
        filename = self.meta[tsv_idx][sample_idx]
        if self.is_tar:
            image_filepath = os.path.join(self.scatter_dir, tsv_idx, filename)
        else:
            image_filepath = os.path.join(self.scatter_dir, tsv_idx, filename[0], filename[1], filename)
        caption_filepath = image_filepath.replace('.jpg', '.txt')
        return image_filepath, caption_filepath

    def get_cumsum(self):
        accum = 0
        cumsum = OrderedDict()
        for tsv_idx, filelist in self.meta.items():
            accum += len(filelist)
            cumsum[tsv_idx] = accum
        self.tsv_indices = list(cumsum.keys())
        self.cumsum_np = np.asarray(list(cumsum.values()))
        return cumsum

    def search_cumsum(self, index):
        tsv_idx = np.searchsorted(self.cumsum_np, index, side="right")
        return self.tsv_indices[tsv_idx]

    def parse_index(self, index):
        tsv_idx = self.search_cumsum(index)
        sample_idx = index - (self.cumsum[tsv_idx] - len(self.meta[tsv_idx]))
        return tsv_idx, sample_idx

    def get_item_with_index(self, index):
        return index, self[index]
