from typing import Any

import torch

from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoTokenizer

from engine.validator import BaseValidator
from data.dataset import build_flickr8k_dataset, build_dataloader


class CaptionValidator(BaseValidator):

    def build_dataset(self, data_path, mode="train"):
        return build_flickr8k_dataset(data_path, self.args.imgsz, self.args.max_len, self.tokenizer, mode)

    def setup(self, stage: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path, use_fast=True)

        self.val_dataset = self.build_dataset(self.val_set, 'val')

    def val_dataloader(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        self.val_loader = build_dataloader(self.val_dataset, self.batch_size * 2,
                                           workers=self.args.workers,
                                           shuffle=False,
                                           persistent_workers=True)
        return self.val_loader

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""

        _, preds = torch.max(preds, dim=2)
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        return preds

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        将 dataloader 的 collate_fn 放在这里
        :param batch:
        :param dataloader_idx:
        :return:
        """
        images = batch[0]
        captions = batch[1]

        dtype = images[0].dtype
        device = images[0].device
        c, _, _ = images[0].shape
        b = len(images)
        max_len = self.args.max_len
        batch_shape = [b, c, self.args.imgsz, self.args.imgsz]

        pad_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        pad_captions = torch.full((b, max_len + 2),
                                  fill_value=self.train_dataset.pad_id,
                                  dtype=torch.long,
                                  device=device)

        for i, (img, cap, pad_tensor, pad_cap) in enumerate(zip(images, captions, pad_tensors, pad_captions)):
            c, h, w = img.shape
            cap_len = len(cap)
            pad_tensor[: c, : h, : w].copy_(img)
            # --------- 开始标记 ------------
            pad_cap[0] = self.train_dataset.cls_id
            pad_cap[1:cap_len + 1].copy_(cap)
            # --------- 结束标记 ------------
            pad_cap[cap_len + 1] = self.train_dataset.sep_id

        batch[0] = pad_tensors
        batch[1] = pad_captions
        batch[2] = torch.tensor(batch[2])[:, None] + 2
        return batch
