import os
import cv2
from skimage import io

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import albumentations as A

from data import augment

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def imread(path):
    return cv2.cvtColor(io.imread(path), cv2.COLOR_RGB2BGR)


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_path, imgsz, max_len, tokenizer, transforms):
        """
        :param data_path: folder where data files are stored
        :param imgsz: image size
        :param max_len: token length
        :param tokenizer: text split
        :param transforms: image transform pipeline
        """
        self.imgsz = imgsz
        self.data_path = data_path
        self.images = []
        self.captions = []
        self.masks = []

        self.cls_id = tokenizer.cls_token_id
        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id

        with open(data_path, 'r', encoding='utf8') as fp:
            lines = fp.readlines()

        for line in lines:
            image_path, captions = line.strip().split('\t')
            captions = captions.split('<sep>')[:-1]
            tokens = tokenizer.batch_encode_plus(captions,
                                                 padding='max_length',
                                                 max_length=max_len,
                                                 add_special_tokens=True,
                                                 return_attention_mask=True,
                                                 return_token_type_ids=False)
            self.images.append(image_path)
            self.captions.append(tokens['input_ids'])
            self.masks.append(tokens['attention_mask'])

        self._transforms = transforms
        self._resize = augment.LongestMaxSize(imgsz)
        self._normalize = augment.Normalize()

    def __getitem__(self, i):
        image_path = self.images[i]
        caption = self.captions[i]
        mask = self.masks[i]
        image = imread(image_path)
        batch = self._resize(image=image)

        if self._transforms is not None:
            batch = self._transforms(**batch)

        return self._normalize(**batch), caption, mask, len(mask)

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


def build_flickr8k_dataset(data_path, imgsz, max_len, tokenizer, mode='train'):
    T = [
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.0),
        A.CLAHE(p=0.01),
        A.RandomBrightnessContrast(p=0.0),
        A.RandomGamma(p=0.0),
        A.ImageCompression(quality_range=(75, 100), p=0.0),
    ]

    transform = A.Compose(T, p=0.0)

    return CaptionDataset(data_path,
                          imgsz,
                          max_len,
                          tokenizer,
                          transforms=transform if mode == 'train' else None)


def build_dataloader(dataset,
                     batch,
                     workers=3,
                     shuffle=False,
                     persistent_workers=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers

    return DataLoader(dataset=dataset,
                      batch_size=batch,
                      shuffle=shuffle,
                      num_workers=nw,
                      pin_memory=PIN_MEMORY,
                      collate_fn=collate_fn,
                      drop_last=True,
                      persistent_workers=persistent_workers)
