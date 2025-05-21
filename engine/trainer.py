from omegaconf import OmegaConf

import torch

import lightning as L
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from utils.lightning_utils import LitProgressBar

from engine.utils import ip_load
from utils.torch_utils import smart_optimizer, ModelEMA, smart_distribute, select_device


class BaseTrainer(LightningModule):
    def __init__(self, cfg):
        super(BaseTrainer, self).__init__()

        self.data = None
        self.train_set = None
        self.val_set = None
        self.args = cfg
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs

        self.save_hyperparameters(self.args)

    def _setup_trainer(self):
        self.model.train()
        self.model.requires_grad_(True)

        device = select_device(self.args.device, self.batch_size)

        accelerator = self.args.device if self.args.device in ["cpu", "tpu", "ipu", "hpu", "mps"] else 'gpu'

        checkpoint_callback = ModelCheckpoint(filename='best',
                                              save_last=True,
                                              monitor='fitness',
                                              mode='max',
                                              auto_insert_metric_name=False,
                                              enable_version_counter=False)

        checkpoint_callback.FILE_EXTENSION = '.pt'

        progress_bar_callback = LitProgressBar(3)

        tensorboard_logger = TensorBoardLogger(save_dir=f'./{self.args.project}/{self.args.task}', name=self.args.name)
        version = tensorboard_logger._get_next_version()

        self.lightning_trainer = L.Trainer(
            accelerator=accelerator,
            devices=device,
            num_nodes=self.args.num_nodes,
            logger=[tensorboard_logger,
                    CSVLogger(save_dir=f'./{self.args.project}/{self.args.task}', name=self.args.name,
                              version=version)],
            strategy=smart_distribute(self.args.num_nodes, device, ip_load(), "8888", "0"),
            max_epochs=self.args.epochs,
            accumulate_grad_batches=max(round(self.args.nbs / self.batch_size), 1),
            gradient_clip_val=10,
            gradient_clip_algorithm="norm",
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            callbacks=[progress_bar_callback, checkpoint_callback]
        )

    def fit(self):
        self._setup_trainer()
        self.train_set, self.val_set = self.get_dataset()
        self.lightning_trainer.fit(self, ckpt_path=self.args.model if self.args.resume else None)

    def get_dataset(self):
        self.data = OmegaConf.load(self.args.data)
        return self.data['train'], self.data.get('val')

    def configure_optimizers(self) -> OptimizerLRScheduler:
        accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * accumulate / self.args.nbs
        optimizer = smart_optimizer(self,
                                    self.args.optimizer,
                                    self.args.lr0,
                                    self.args.momentum,
                                    weight_decay)

        self.lr_lambda = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      last_epoch=self.current_epoch - 1,
                                                      lr_lambda=self.lr_lambda)

        return [optimizer], [scheduler]
