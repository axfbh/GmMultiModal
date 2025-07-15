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
        masks = batch[2]
        cap_ids = batch[3]

        dtype = images[0].dtype
        device = images[0].device
        c, _, _ = images[0].shape
        b = len(images)

        res_tensors = []
        res_caps = []
        res_masks = []
        res_ids = []

        for i in range(b):
            img = images[i]
            cap = captions[i]
            mask = masks[i]
            cap_id = cap_ids[i]
            cpi_shape = [cap_id, c, self.args.imgsz, self.args.imgsz]
            pad_tensor = torch.zeros(cpi_shape, dtype=dtype, device=device)
            c, h, w = img.shape
            pad_tensor[:, :c, : h, : w].copy_(img)
            res_tensors.append(pad_tensor)
            res_caps.append(torch.as_tensor(cap))
            res_masks.append(torch.as_tensor(mask))
            res_ids.append(torch.ones(cap_id, dtype=torch.long, device=device) * i)

        batch[0] = torch.cat(res_tensors)
        batch[1] = torch.cat(res_caps)
        batch[2] = torch.cat(res_masks).sum(-1, keepdim=True)
        batch[3] = torch.cat(res_ids)
        return tuple(batch)
