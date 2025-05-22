import os
import h5py
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.data_folder = data_folder
        self.data_name = data_name
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # # Open hdf5 file where images are stored
        # self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        # self.imgs = self.h['images']
        #
        # # Captions per image
        # self.cpi = self.h.attrs['captions_per_image']

        # Open hdf5 file where images are stored
        with h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r') as f:
            self.imgs = f['images'][:]

            # Captions per image
            self.cpi = f.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        np_img = self.imgs[i // self.cpi]
        img = torch.FloatTensor(np_img / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions, np_img

    def __len__(self):
        return self.dataset_size


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


def build_flickr8k_dataset(data_folder, data_name, mode='TRAIN'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return CaptionDataset(data_folder, data_name, mode, normalize)


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
