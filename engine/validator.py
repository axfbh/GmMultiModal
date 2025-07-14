from omegaconf import OmegaConf

import lightning as L
from lightning import LightningModule

from utils.lightning_utils import LitProgressBar
from torchmetrics.text import BLEUScore


class BaseValidator(LightningModule):
    def __init__(self, cfg=None):
        super(BaseValidator, self).__init__()
        self.args = cfg

        self.tokenizer = None
        self.evaluator = None

        self.data = None
        self.val_set = None
        self.val_dataset = None
        self.val_loader = None

        self.lightning_validator = None

        self.batch_size = None if self.args is None else int(self.args.batch * 0.5)

    def _setup_validator(self):
        self.model.eval()

        device = [0]

        accelerator = self.args.device if self.args.device in ["cpu", "tpu", "ipu", "hpu", "mps"] else 'gpu'

        progress_bar_callback = LitProgressBar(10)

        self.lightning_validator = L.Trainer(
            accelerator=accelerator,
            devices=device,
            num_nodes=1,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            callbacks=[progress_bar_callback]
        )

    def validate(self):
        self._setup_validator()
        self.val_set = self.get_dataset()
        self.lightning_validator.validate(self)

    def get_dataset(self):
        self.data = OmegaConf.load(self.args.data)
        return self.data['val']

    def configure_model(self) -> None:
        self.model.device = self.device
        self.model.args = self.args

    def on_validation_start(self):
        self.evaluator = BLEUScore()

    def forward(self, batch):
        return self.model(batch)

    def validation_step(self, batch, batch_idx):
        preds, targets = self.ema.ema(batch) if hasattr(self, 'ema') else self(batch)
        preds = self.postprocess(preds)
        targets = self.tokenizer.batch_decode(targets, skip_special_tokens=True)
        targets = [[t] for t in targets]
        self.evaluator.update(preds, targets)

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        return preds
