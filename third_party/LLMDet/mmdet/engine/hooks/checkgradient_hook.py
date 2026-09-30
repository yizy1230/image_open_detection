from mmengine.hooks import Hook
from mmengine.runner import Runner

from typing import Optional

from mmdet.registry import HOOKS

@HOOKS.register_module()
class GradientCheckHook(Hook):
    def after_train_iter(self,
            runner: Runner,
            batch_idx: int,
            data_batch: Optional[dict] = None,
            outputs: Optional[dict] = None) -> None:
        # Get the model
        model = runner.model
        # Walk parameters and print gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"{name} gradient norm: {param.grad.norm()}")
