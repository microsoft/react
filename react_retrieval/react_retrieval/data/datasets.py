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

    def load_zipfile(self, zipfile_name):
        full_path = os.path.join(self.data_root, zipfile_name)  #  수정됨: 명확한 경로 처리
        if full_path not in self.zipfiles:
            try:
                self.zipfiles[full_path] = ZipFile(full_path)
            except BadZipFile as e:
                logging.error(f"Failed to open zipfile: {full_path}")  #  개선됨: 로깅 추가
                raise e
        return self.zipfiles[full_path]

    def read_image(self, index):
        img_info = self.infolist[index]
        zipfile_name, imagefile = img_info['img_path'].split('@')
        zipfile = self.load_zipfile(zipfile_name)

        try:
            with zipfile.open(imagefile) as img_file:
                image = Image.open(img_file).convert('RGB')  #  개선됨: with-context 사용
        except KeyError:
            logging.error(f"Image file {imagefile} not found in zip archive {zipfile_name}")  # ✅ 개선됨
            raise
        except BadZipFile:
            raise RuntimeError(f"Bad zip file in reading {img_info['img_path']}")  #  개선됨

        return image

    def __getitem__(self, index):
        image = self.read_image(index)
        return self.transform(image) if self.transform else image  #  개선됨: 간결화

class TSVDataset(data.Dataset):
    def __init__(self, file_name, transform=None):
        super().__init__()
        self.tsv_file = TSVFile(file_name)
        self.transform = transform

    def __len__(self):
        return len(self.tsv_file)

    def __getitem__(self, index):
        item = self.tsv_file[index]
        return self.transform(item) if self.transform else item  #  개선됨

class PairsDataset(data.Dataset):
    def __init__(self, image_file_name, text_file_name, image_transform=None, text_transform=None):
        super().__init__()
        self.image_dataset = TSVDataset(image_file_name, image_transform)
        self.text_dataset = TSVDataset(text_file_name, text_transform)

        if len(self.image_dataset) != len(self.text_dataset):  #  개선됨: 명확한 에러 처리
            raise ValueError("Image and text datasets must be of equal length.")

    def __len__(self):
        return len(self.image_dataset)

    def get_image(self, index):
        raw_image_data = self.image_dataset.tsv_file[index]
        return Image.open(BytesIO(base64.b64decode(raw_image_data[1]))).convert('RGB')

    def get_image_raw(self, index):
        return self.image_dataset.tsv_file[index][1]

    def get_text(self, index):
        raw_text_data = self.text_dataset.tsv_file[index]
        captions = json.loads(raw_text_data[1]).get('captions', [""])  #  개선됨: get 사용
        return captions[0] if captions else ""

    def __getitem__(self, index):
        image_filename, image = self.image_dataset[index]
        text_filename, text = self.text_dataset[index]

        if image_filename != text_filename:  #  개선됨: assert 대신 명시적 예외 처리
            raise ValueError(f"Filename mismatch: {image_filename} != {text_filename}")

        return image, text, {
            'index': index,
            'filename': image_filename,
        }

def decode_image(image_item, fn):
    try:
        return image_item[0], fn(Image.open(BytesIO(base64.b64decode(image_item[1]))).convert('RGB'))  # ✅ 개선됨: try-catch
    except Exception as e:
        logging.error(f"Failed to decode image: {e}")  #  개선됨
        raise

def decode_text(text_item):
    try:
        captions = json.loads(text_item[1]).get('captions', [""])  #  개선됨
        text_captions_first = captions[0] if captions and captions[0] else ""
        if not text_captions_first:
            logging.warning(f"Found null or empty caption in file {text_item[0]}, using empty string.")  # ✅ 개선됨
        texts = clip.tokenize([text_captions_first], context_length=77, truncate=True)
        return text_item[0], texts.squeeze()
    except Exception as e:
        logging.error(f"Failed to decode text: {e}")  #  개선됨
        raise

def encode_as_string(arr):
    if not isinstance(arr, np.ndarray):  #  개선됨
        arr = arr.data.cpu().numpy()
    return base64.b64encode(arr.tobytes()).decode('utf-8')

def decode_pairs_feature(item):
    try:
        index, filename, image_feature, text_feature = item
        index = int(index)
        image_feature = np.frombuffer(base64.b64decode(image_feature), dtype='float16')
        text_feature = np.frombuffer(base64.b64decode(text_feature), dtype='float16')
        return index, filename, image_feature, text_feature
    except Exception as e:
        logging.error(f"Failed to decode pair features: {e}")  #  개선됨
        raise
