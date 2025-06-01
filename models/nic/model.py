from engine.model import Model
from models.nic.caption.train import CaptionTrainer
from models.nic.modules import Nica, Nic


class NIC(Model):
    def __init__(self, model="nic.pt", task=None):
        super().__init__(model=model, task=task)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "caption": {
                "model": {
                    'nic': Nic,
                    'nica': Nica,
                },
                "trainer": CaptionTrainer,
            },
        }
