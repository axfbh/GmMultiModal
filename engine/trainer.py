from typing import Any, Dict
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

        self.last_opt_step = None
        self.ema = None

        self.data = None
        self.train_set = None
        self.train_dataset = None
        self.train_loader = None
        self.val_set = None
        self.val_dataset = None

        self.lr_lambda = None
        self.lightning_trainer = None

        self.args = cfg
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs

        self.automatic_optimization = False
        self.accumulate_grad_batches = max(round(self.args.nbs / self.batch_size), 1)

        self.save_hyperparameters(self.args)

    def _setup_trainer(self):
        device = select_device(self.args.device, self.batch_size)

        accelerator = self.args.device if self.args.device in ["cpu", "tpu", "ipu", "hpu", "mps"] else 'gpu'

        checkpoint_callback = ModelCheckpoint(filename='best',
                                              save_last=True,
                                              monitor='fitness',
                                              mode='max',
                                              auto_insert_metric_name=False,
                                              enable_version_counter=False)

        checkpoint_callback.FILE_EXTENSION = '.pt'

        progress_bar_callback = LitProgressBar(100)

        csv_logger = CSVLogger(save_dir=f'./{self.args.project}/{self.args.task}', name=self.args.name)
        version = csv_logger._get_next_version()

        tensorboard_logger = TensorBoardLogger(save_dir=f'./{self.args.project}/{self.args.task}',
                                               name=self.args.name,
                                               version=version)

        self.lightning_trainer = L.Trainer(
            accelerator=accelerator,
            devices=device,
            num_nodes=self.args.num_nodes,
            logger=[csv_logger, tensorboard_logger],
            strategy=smart_distribute(self.args.num_nodes, device, ip_load(), "8888", "0"),
            max_epochs=self.args.epochs,
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
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.decoder.parameters()),
                                             lr=1e-4)

        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.decoder.parameters()),
                                             lr=4e-4)

        encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=8, gamma=0.8)
        decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=8, gamma=0.8)

        return [encoder_optimizer, decoder_optimizer], [encoder_scheduler, decoder_scheduler]

    def configure_model(self) -> None:
        self.model.device = self.device
        self.model.args = self.args

    def on_train_start(self) -> None:
        self.ema = ModelEMA(self.model, updates=self.args.updates)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        encoder_optimizer, decoder_optimizer = self.optimizers()
        encoder_scheduler, decoder_scheduler = self.lr_schedulers()

        loss, loss_dict = self(batch)

        # loss = loss * self.accumulate_grad_batches * self.trainer.world_size

        self.log_dict(loss_dict,
                      on_step=True,
                      on_epoch=True,
                      sync_dist=True,
                      prog_bar=True,
                      rank_zero_only=True,
                      batch_size=self.batch_size)

        decoder_optimizer.zero_grad()
        self.manual_backward(loss)

        self.clip_gradients(decoder_optimizer, 5., "value")

        decoder_optimizer.step()
        decoder_scheduler.step()
        self.ema.update(self.model)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['ema'] = self.ema.ema
        checkpoint['updates'] = self.ema.updates
        checkpoint['state_dict'] = self.ema.ema.state_dict()

    def load_state_dict(self, *args, **kwargs):
        """
           模型参数，改在外部加载
        """
        pass
