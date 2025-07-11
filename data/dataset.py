import os
import cv2
from skimage import io

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import augment

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def imread(path):
    return cv2.cvtColor(io.imread(path), cv2.COLOR_RGB2BGR)


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_path, imgsz, transforms):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.imgsz = imgsz
        self.data_path = data_path
        self.data_set = []
        with open(data_path, 'r', encoding='utf8') as fp:
            lines = fp.readlines()

        for l in lines:
            l = l.strip()
            image_path, caption = l.split('\t')
            self.data_set.append([image_path, caption])

        self._transforms = transforms
        self._resize = A.LongestMaxSize(self.imgsz, p=1)
        self._normalize = A.Normalize(p=1)
        self._totensor = A.ToTensorV2()

    def __getitem__(self, i):
        image_path, caption = self.data_set[i]
        image = imread(image_path)
        batch = self._resize(image=image)

        if self._transforms is not None:
            batch = self._transforms(**batch)

        batch = self._normalize(**batch)
        return self._totensor(**batch)['image'], caption

        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        # np_img = self.imgs[i // self.cpi]
        # img = torch.FloatTensor(np_img / 255.)
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # caption = torch.LongTensor(self.captions[i])
        #
        # caplen = torch.LongTensor([self.caplens[i]])
        #
        # if self.split == 'TRAIN':
        #     return img, caption, caplen
        # else:
        #     # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
        #     all_captions = torch.LongTensor(
        #         self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
        #     return img, caption, caplen, all_captions, np_img

    def __len__(self):
        return len(self.data_set)


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


def build_flickr8k_dataset(data_path, imgsz, mode='train'):
    T = [
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.0),
        A.CLAHE(p=0.01),
        A.RandomBrightnessContrast(p=0.0),
        A.RandomGamma(p=0.0),
        A.ImageCompression(quality_range=(75, 100), p=0.0),
    ]

    transform = A.Compose(T)

    return CaptionDataset(data_path,
                          imgsz,
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
