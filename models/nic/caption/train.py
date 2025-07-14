from typing import Any

import torch

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from transformers import AutoTokenizer

from engine.trainer import BaseTrainer

from data.dataset import build_flickr8k_dataset, build_dataloader
from models.nic.caption.val import CaptionValidator


# 先执行 BaseTrainer，从 BaseTrainer super 跳到执行 DetectionValidator
# 因此 DetectionValidator 创建的重复信息会被，后续执行BaseTrainer覆盖，不影响训练时候的参数
class CaptionTrainer(BaseTrainer, CaptionValidator):

    def build_dataset(self, data_path, mode="train"):
        return build_flickr8k_dataset(data_path, self.args.imgsz, self.args.max_len, self.tokenizer, mode)

    def setup(self, stage: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path, use_fast=True)

        self.train_dataset = self.build_dataset(self.train_set, 'train')

        if self.val_set is not None:
            self.val_dataset = self.build_dataset(self.val_set, 'val')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.train_loader = build_dataloader(self.train_dataset, self.batch_size,
                                             workers=self.args.workers,
                                             shuffle=True,
                                             persistent_workers=True)
        return self.train_loader

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        将 dataloader 的 collate_fn 放在这里
        :param batch:
        :param dataloader_idx:
        :return:
        """
        images = batch[0]
        captions = batch[1]
        masks = batch[2]

        dtype = images[0].dtype
        device = images[0].device
        c, _, _ = images[0].shape
        b = len(images)

        res_tensors = []
        res_caps = []
        res_masks = []

        for i in range(b):
            img = images[i]
            cap = captions[i]
            mask = masks[i]
            cpi = len(mask)
            cpi_shape = [cpi, c, self.args.imgsz, self.args.imgsz]
            pad_tensor = torch.zeros(cpi_shape, dtype=dtype, device=device)
            c, h, w = img.shape
            pad_tensor[:, :c, : h, : w].copy_(img)
            res_tensors.append(pad_tensor)
            res_caps.append(torch.as_tensor(cap))
            res_masks.append(torch.as_tensor(mask))

        batch[0] = torch.cat(res_tensors)
        batch[1] = torch.cat(res_caps)
        batch[2] = torch.cat(res_masks).sum(-1, keepdim=True)
        return batch
