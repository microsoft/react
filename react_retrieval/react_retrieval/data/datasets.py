from io import BytesIO
import base64
from PIL import Image
import json
import logging

from torch.utils import data
import clip
import numpy as np

from .tsv import TSVFile

import os
from zipfile import ZipFile, BadZipFile


class ICinWJsonDataset(data.Dataset):
    def __init__(self, data_root, infolist, transform=None):
        super().__init__()

        logging.info(f'Initializing ICinW JSON dataset with {infolist}')
        with open(infolist, 'r') as fp:
            self.infolist = json.load(fp)
        self.data_root = data_root
        self.zipfiles = {}
        self.transform = transform

    def __len__(self):
        return len(self.infolist)

    def load_zipfile(self, zipfile):
        zipfile = os.path.join(self.data_root, zipfile)
        if zipfile not in self.zipfiles:
            self.zipfiles[zipfile] = ZipFile(zipfile)
        return self.zipfiles[zipfile]

    def read_image(self, index):
        img_info = self.infolist[index]
        zipfile, imagefile = img_info['img_path'].split('@')
        zipfile = self.load_zipfile(zipfile)

        try:
            image = Image.open(BytesIO(zipfile.read(imagefile))).convert('RGB')
        except BadZipFile:
            assert False, f"bad zip file in reading {img_info['img_path']}"

        return image

    def __getitem__(self, index):
        image = self.read_image(index)
        if self.transform is not None:
            return self.transform(image)
        return image


class TSVDataset(data.Dataset):
    def __init__(self, file_name, transform=None):
        super().__init__()

        self.tsv_file = TSVFile(file_name)
        self.transform = transform
    
    def __len__(self):
        return len(self.tsv_file)

    def __getitem__(self, index):
        item = self.tsv_file[index]
        if self.transform is not None:
            return self.transform(item)
        return item


class PairsDataset(data.Dataset):
    def __init__(self, image_file_name, text_file_name, image_transform=None, text_transform=None):
        super().__init__()

        self.image_dataset = TSVDataset(image_file_name, image_transform)
        self.text_dataset = TSVDataset(text_file_name, text_transform)

        assert len(self.image_dataset) == len(self.text_dataset)
    
    def __len__(self):
        return len(self.image_dataset)

    def get_image(self, index):
        raw_image_data = self.image_dataset.tsv_file[index]
        return Image.open(BytesIO(base64.b64decode(raw_image_data[1]))).convert('RGB')

    def get_image_raw(self, index):
        raw_image_data = self.image_dataset.tsv_file[index]
        return raw_image_data[1]

    def get_text(self, index):
        raw_text_data = self.text_dataset.tsv_file[index]
        return json.loads(raw_text_data[1])['captions'][0]

    def __getitem__(self, index):
        image_filename, image = self.image_dataset[index]
        text_filename, text = self.text_dataset[index]

        assert image_filename == text_filename

        return image, text, {
            'index': index,
            'filename': image_filename,
        }


def decode_image(image_item, fn):
    return image_item[0], fn(Image.open(BytesIO(base64.b64decode(image_item[1]))).convert('RGB'))


def decode_text(text_item):
    text_captions_first = json.loads(text_item[1])['captions'][0]
    if text_captions_first is None:
        text_captions_first = ""
        print(f'Found null caption in file {text_item[0]}, using empty string.')
    texts = clip.tokenize([text_captions_first], context_length=77, truncate=True)
    return text_item[0], texts.squeeze()


def encode_as_string(arr):
    if type(arr) != np.ndarray:
        arr = arr.data.cpu().numpy()
    return base64.b64encode(arr.tobytes()).decode('utf-8')


def decode_pairs_feature(item):
    index, filename, image_feature, text_feature = item
    index = int(index)
    image_feature = np.frombuffer(base64.b64decode(image_feature), dtype='float16')
    text_feature = np.frombuffer(base64.b64decode(text_feature), dtype='float16')
    return index, filename, image_feature, text_feature
