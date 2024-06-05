from mmaction.registry import METRICS
from mmengine.evaluator import BaseMetric
from typing import Any, Optional, Sequence
from copy import deepcopy


@METRICS.register_module()
class LossRestore(BaseMetric):
    default_prefix: Optional[str] = 'loss'

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        data_samples = deepcopy(data_samples)
        for data_sample in data_samples:
            loss = data_sample['restore_loss']['item'].detach().cpu().item()
            self.results.append(loss)
    
    def compute_metrics(self, results: list) -> dict:
        print('loss/loss_restore', sum(results) / len(results))
        return {'loss_restore': sum(results) / len(results)}

    