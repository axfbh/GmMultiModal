import os, sys
from typing import Any, Dict
import subprocess
import signal

from omegaconf import OmegaConf

import torch

import lightning as L
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from utils.lightning_utils import LitProgressBar

from engine.utils import ip_load
from utils.torch_utils import smart_optimizer, ModelEMA, smart_distribute, select_device, is_frozen


class BaseTrainer(LightningModule):
    def __init__(self, cfg):
        super(BaseTrainer, self).__init__()

        self.ema = None

        self.is_decoder = True
        self.is_encoder = True
        self.encoder_optimizer = None
        self.decoder_scheduler = None
        self.encoder_scheduler = None
        self.decoder_optimizer = None
        self.lr_lambda = None

        self.lightning_trainer = None

        self.args = cfg
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs

        self.data = None
        self.train_set, self.val_set = self.get_dataset()
        self.train_dataset = None
        self.train_loader = None
        self.val_dataset = None

        self.automatic_optimization = False

        self.save_hyperparameters(self.args)

    def _start_tensorboard(self):
        port = 6006
        # 训练开始后启动 TensorBoard 进程
        cmd = f"tensorboard --logdir=./{self.args.project}/{self.args.task}/{self.args.name} --port={port}"
        self.cmd_process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True if os.name == 'nt' else False  # Windows 需要 shell=True
        )
        print(f"\nTensorBoard 已启动：http://localhost:{port}\n")

    def _close_tensorboard(self):
        # 训练结束后终止 TensorBoard 进程
        if self.cmd_process:
            if sys.platform == 'win32':
                self.cmd_process.send_signal(signal.CTRL_BREAK_EVENT)  # Windows 用 Ctrl+Break
            else:
                self.cmd_process.send_signal(signal.SIGINT)  # Unix 用 SIGINT (Ctrl+C)
            self.cmd_process.wait()
            print("TensorBoard 进程已终止")

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

        progress_bar_callback = LitProgressBar(20)

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
        self.lightning_trainer.fit(self, ckpt_path=self.args.model if self.args.resume else None)

    def get_dataset(self):
        self.data = OmegaConf.load(self.args.data)
        return self.data['train'], self.data.get('val')

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim, sche = [], []

        accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * accumulate / self.args.nbs
        self.lr_lambda = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf

        if self.is_encoder:
            # self.encoder_optimizer = smart_optimizer(self.model.encoder,
            #                                          self.args.optimizer,
            #                                          self.args.lre0,
            #                                          self.args.momentum,
            #                                          weight_decay)
            # self.encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(self.encoder_optimizer,
            #                                                            last_epoch=self.current_epoch - 1,
            #                                                            lr_lambda=self.lr_lambda)
            optim.append(self.encoder_optimizer)
            sche.append(self.encoder_scheduler)

        if self.is_decoder:
            # self.decoder_optimizer = torch.optim.Adam(
            #     params=filter(lambda p: p.requires_grad, self.model.decoder.parameters()),
            #     lr=4e-4)
            # self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=8, gamma=0.8)
            self.decoder_optimizer = smart_optimizer(self.model.decoder,
                                                     self.args.optimizer,
                                                     self.args.lrd0,
                                                     self.args.momentum,
                                                     weight_decay)

            self.decoder_scheduler = torch.optim.lr_scheduler.LambdaLR(self.decoder_optimizer,
                                                                       last_epoch=self.current_epoch - 1,
                                                                       lr_lambda=self.lr_lambda)
            optim.append(self.decoder_optimizer)
            sche.append(self.decoder_scheduler)

        return optim, sche

    def configure_model(self) -> None:
        self.model.device = self.device
        self.model.args = self.args
        self.is_encoder = not is_frozen(self.model.encoder)
        self.is_decoder = not is_frozen(self.model.decoder)
        self.trainer.accumulate_grad_batches = max(round(self.args.nbs / self.batch_size), 1)

    def on_train_start(self) -> None:
        self._start_tensorboard()
        self.ema = ModelEMA(self.model, updates=self.args.updates)

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        epoch = self.current_epoch
        ni = batch_idx + self.batch_size * epoch
        nw = max(round(self.batch_size * self.args.warmup_epochs), 100)

        optimizers = []
        if self.is_decoder:
            optimizers.append(self.decoder_optimizer)
        if self.is_encoder:
            optimizers.append(self.encoder_optimizer)

        if ni <= nw:
            ratio = ni / nw
            interpolated_accumulate = 1 + (self.args.nbs / self.batch_size - 1) * ratio
            self.trainer.accumulate_grad_batches = max(1, round(interpolated_accumulate))
            for optimizer in optimizers:
                for j, param_group in enumerate(optimizer.param_groups):
                    # 学习率线性插值
                    lr_start = self.args.warmup_bias_lr if j == 0 else 0.0
                    lr_end = param_group["initial_lr"] * self.lr_lambda(epoch)
                    param_group["lr"] = lr_start + (lr_end - lr_start) * ratio

                    if "momentum" in param_group:
                        param_group["momentum"] = self.args.warmup_momentum + (
                                self.args.momentum - self.args.warmup_momentum) * ratio

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)

        self.log_dict(loss_dict,
                      on_step=True,
                      on_epoch=True,
                      sync_dist=True,
                      prog_bar=True,
                      rank_zero_only=True,
                      batch_size=self.batch_size)

        # loss = loss / self.trainer.accumulate_grad_batches * self.trainer.world_size
        self.manual_backward(loss)

        # if not self.trainer.fit_loop._should_accumulate():
        self._optimizer_step()

    def _optimizer_step(self) -> None:
        # 统计 optim step 执行次数，即 global_step
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()

        if self.is_decoder:
            self.clip_gradients(self.decoder_optimizer, 5., "value")
            self.decoder_optimizer.step()
            self.decoder_optimizer.zero_grad()

        if self.is_encoder:
            self.clip_gradients(self.encoder_optimizer, 5., "value")
            self.encoder_optimizer.step()
            self.encoder_optimizer.zero_grad()

        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

        self.ema.update(self.model)

    def on_train_epoch_end(self) -> None:
        if self.is_decoder:
            self.decoder_scheduler.step()

        if self.is_encoder:
            self.encoder_scheduler.step()

    def on_train_end(self) -> None:
        self._close_tensorboard()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['ema'] = self.ema.ema
        checkpoint['updates'] = self.ema.updates
        checkpoint['state_dict'] = self.ema.ema.state_dict()

    def load_state_dict(self, *args, **kwargs):
        """
           模型参数，改在外部加载
        """
        pass
