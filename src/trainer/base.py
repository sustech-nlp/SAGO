# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import re
import logging
# from transformers import Trainer
from .hf_mtask_trainer import HfMultiTaskTrainer as Trainer  # Custom Trainer that supports multiple evaluators
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any

logger = logging.getLogger(__name__)


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, trainable_params_regex=[".*"], *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        super().__init__(*args, **kwargs)

        # Unfreeze only the selected parameters
        self.trainable_params_regex = (
            trainable_params_regex  # Regex for selecting params
        )

    def create_optimizer(self):
        if self.trainable_params_regex == [".*"]:
            super().create_optimizer()
        else:
            self._freeze_all_params(self.model, False)
            # This makes the optimizer to select only trainable params
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            super().create_optimizer()
            if self.args.gradient_checkpointing:
                self._freeze_all_params(self.model, True)  # Require for gradient_checkpointing=true

    def _freeze_all_params(self, model, requires_grad=True):
        """Freeze all parameters in the model initially."""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        """Unfreeze specific parameters that match the regex patterns."""
        for name, param in model.named_parameters():
            if any(re.fullmatch(pattern, name) for pattern in trainable_params_regex):
                param.requires_grad = requires_grad
                # print(f"{name}:requires_grad\t{requires_grad}")

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators:
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_metrics = {}
                    for _, evaluator in self.evaluators.items():
                        eval_args = {
                            "output_dir": output_dir,
                            "template_args": self.template_args,
                            "model": self.model,
                            "tokenizer": self.tokenizer,
                        }
                        eval_metrics.update(evaluator.evaluate(**eval_args))
                    self.log(eval_metrics)
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
