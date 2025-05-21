from engine.model import Model
from models.sat.caption.train import CaptionTrainer
from models.sat.modules import Sat


class SAT(Model):
    def __init__(self, model="sat.pt", task=None):
        super().__init__(model=model, task=task)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "caption": {
                "model": {
                    'sat': Sat,
                },
                "trainer": CaptionTrainer,
            },
        }
