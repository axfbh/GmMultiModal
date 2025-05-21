from typing import Any

import torch

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

from engine.trainer import BaseTrainer

from data.flickr8k_dataset import build_flickr8k_dataset, build_dataloader


# from models.yolo.detect.val import DetectionValidator


# 先执行 BaseTrainer，从 BaseTrainer super 跳到执行 DetectionValidator
# 因此 DetectionValidator 创建的重复信息会被，后续执行BaseTrainer覆盖，不影响训练时候的参数
class CaptionTrainer(BaseTrainer):

    def build_dataset(self, img_path, ann_path, mode="train"):
        return build_flickr8k_dataset(img_path, ann_path)

    def setup(self, stage: str) -> None:
        self.train_dataset = self.build_dataset(self.train_set['data_folder'], self.train_set['data_name'], "TRAIN")

        if self.val_set is not None:
            self.val_dataset = self.build_dataset(self.val_set['data_folder'], self.val_set['data_name'], "VAL")

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

        dtype = images[0].dtype
        device = images[0].device
        c, _, _ = images[0].shape
        batch_shape = [len(images), c, self.args.imgsz, self.args.imgsz]
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for i, (img, pad_img) in enumerate(zip(images, tensor)):
            c, h, w = img.shape
            pad_img[: c, : h, : w].copy_(img)

        batch[0] = tensor
        return batch
