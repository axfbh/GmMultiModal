import os
import torch
from torch.utils.data import DataLoader

from torchvision.datasets.flickr import Flickr8k

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


def build_flickr8k_dataset(img_folder, ann_file):
    return Flickr8k(img_folder, ann_file)


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
