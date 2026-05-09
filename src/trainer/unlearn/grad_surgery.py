from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import re
import copy
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
)
from deepspeed.utils import safe_get_full_grad, safe_set_full_grad, safe_get_local_grad, safe_set_local_grad

from .grad_diff import GradDiff
from .npo import NPO
from .simnpo import SimNPO
from .rmu import RMU


class GradSurgeryMixin:
    """
    Gradient Surgery Mixin:
    1. SAGO (default): Always keep the retain gradient when conflicts occur, but use only forget gradient for non-conflicts
    2. SAGO_Prefer_Retain: Always keep the retain gradient when conflicts occur
    3. PCGrad: Apply PCGrad projection to forget gradients parameter-wise, then combine normally with retain gradients
    4. PCGrad_Global: Apply global PCGrad projection considering all parameters' gradients at once
    5. BLUR: Apply global projection to retain gradients (project retain onto forget direction and subtract)
    6. Sum: Simply sum the forget and retain gradients with their respective weights
    
    Notes:
    - By default, losses are (forget, retain) like GradDiff and combined with weights (gamma, alpha).
    - Not compatible with FSDP or SageMaker Model Parallel.
    """

    def __init_grad_surgery__(
        self,
        surgery_strategy: str = "sago",  # ["sago", "sago_prefer_retain", "pcgrad", "pcgrad_global", "blur", "sum"]
        **kwargs,
    ):
        self.surgery_strategy = surgery_strategy
        self._accum_forget = None
        self._accum_retain = None

        # Fail fast on unsupported distributed engines
        if is_sagemaker_mp_enabled():
            raise RuntimeError(
                "GradSurgery does not support SageMaker Model Parallel. Disable SMP and use DDP instead."
            )
        if getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_fsdp_xla_enabled", False) or getattr(self, "is_fsdp_xla_v2_enabled", False):
            raise RuntimeError(
                "GradSurgery does not support FSDP. Disable FSDP and use DDP instead."
            )

    def _detect_conflicts(self, forget_grad: torch.Tensor, retain_grad: torch.Tensor) -> torch.Tensor:
        """
        Detect gradient sign conflicts.
        Returns a boolean mask where True indicates a conflict (opposite signs).
        """
        # Consider zero gradients as non-conflicting
        forget_sign = torch.sign(forget_grad)
        retain_sign = torch.sign(retain_grad)
        
        # Conflict occurs when signs are opposite (one positive, one negative)
        conflicts = (forget_sign * retain_sign) < 0
        return conflicts

    def _apply_sago(self, forget_grad: torch.Tensor, retain_grad: torch.Tensor, conflicts: torch.Tensor) -> torch.Tensor:
        """
        SAGO (Sign-Aligned Gradient Optimization) - Main method
        Always keep the retain gradient when conflicts occur, but use only forget gradient for non-conflicts.
        """
        # For conflicting gradients, use only retain gradient
        # For non-conflicting gradients, use only forget gradient
        non_conflicts = ~conflicts
        
        result = torch.zeros_like(forget_grad)
        result[conflicts] = self.alpha * retain_grad[conflicts]
        result[non_conflicts] = self.gamma * forget_grad[non_conflicts]
        
        return result

    def _apply_sago_prefer_retain(self, forget_grad: torch.Tensor, retain_grad: torch.Tensor, conflicts: torch.Tensor) -> torch.Tensor:
        """
        SAGO_Prefer_Retain - Variant method
        Always keep the retain gradient when conflicts occur.
        """
        # For conflicting gradients, use only retain gradient
        # For non-conflicting gradients, use weighted sum as usual
        non_conflicts = ~conflicts
        
        result = torch.zeros_like(forget_grad)
        result[conflicts] = self.alpha * retain_grad[conflicts]
        result[non_conflicts] = self.gamma * forget_grad[non_conflicts] + self.alpha * retain_grad[non_conflicts]
        
        return result

    def _apply_pcgrad(self, forget_grad: torch.Tensor, retain_grad: torch.Tensor) -> torch.Tensor:
        """
        PCGrad
        Apply PCGrad projection to forget gradients, then combine with retain gradients normally.
        """
        eps = 1e-10
        
        # Calculate dot product between forget and retain gradients
        dot_product = (forget_grad * retain_grad).sum()
        
        # Apply PCGrad projection only if there's a conflict (negative dot product)
        if (dot_product < 0).item():
            # Calculate retain gradient squared norm
            retain_norm_sq = (retain_grad * retain_grad).sum()
            
            # Calculate projection coefficient
            coeff = dot_product / (retain_norm_sq + eps)
            
            # Apply projection: g1_proj = g1 - coeff * g2
            projected_forget_grad = forget_grad - coeff * retain_grad
        else:
            # No conflict, use original forget gradient
            projected_forget_grad = forget_grad
        
        # Combine projected forget gradient with retain gradient using weighted sum
        return self.gamma * projected_forget_grad + self.alpha * retain_grad

    def _apply_pcgrad_global(self, forget_grads: List[torch.Tensor], retain_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        PCGrad Global
        Apply global PCGrad projection to all forget gradients, then combine with retain gradients normally.
        
        Args:
            forget_grads: List of forget gradients for all parameters
            retain_grads: List of retain gradients for all parameters
            
        Returns:
            List of combined gradients for all parameters
        """
        eps = 1e-10
        
        global_dot_product = self._dot_product(forget_grads, retain_grads)
        
        # Apply global PCGrad projection only if there's a global conflict (negative dot product)
        if (global_dot_product < 0).item():
            # Calculate global retain gradient squared norm
            global_retain_norm_sq = self._dot_product(retain_grads, retain_grads)
            
            # Calculate global projection coefficient
            global_coeff = global_dot_product / (global_retain_norm_sq + eps)
            
            # Apply projection to all forget gradients: g1_proj = g1 - coeff * g2
            projected_forget_grads = [f_grad - global_coeff * r_grad for f_grad, r_grad in zip(forget_grads, retain_grads)]
        else:
            # No global conflict, use original forget gradients
            projected_forget_grads = forget_grads
        
        # Combine projected forget gradients with retain gradients using weighted sum
        return [self.gamma * proj_f_grad + self.alpha * r_grad  for proj_f_grad, r_grad in zip(projected_forget_grads, retain_grads)]

    def _apply_blur(self, forget_grads: List[torch.Tensor], retain_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            forget_grads: List of forget gradients for all parameters
            retain_grads: List of retain gradients for all parameters
            
        Returns:
            List of combined gradients for all parameters
        """
        eps = 1e-10
        
        global_dot_product = self._dot_product(retain_grads, forget_grads)
        
        # Calculate global forget gradient squared norm
        global_forget_norm_sq = self._dot_product(forget_grads, forget_grads)
        
        # Calculate global projection coefficient
        global_coeff = global_dot_product / (global_forget_norm_sq + eps)
        
        # Apply projection to all retain gradients: r_proj = r - coeff * f
        projected_retain_grads = [r_grad - global_coeff * f_grad for f_grad, r_grad in zip(forget_grads, retain_grads)]
        
        # Combine forget gradients with projected retain gradients using weighted sum
        return [self.gamma * f_grad + self.alpha * proj_r_grad for f_grad, proj_r_grad in zip(forget_grads, projected_retain_grads)]

    def _apply_sum(self, forget_grad: torch.Tensor, retain_grad: torch.Tensor) -> torch.Tensor:
        """
        Sum strategy
        Simply sum the forget and retain gradients with their respective weights.
        """
        return self.gamma * forget_grad + self.alpha * retain_grad

    def _apply_surgery_strategy(self, forget_grad: torch.Tensor, retain_grad: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified surgery strategy to resolve gradient conflicts.
        """
        conflicts = self._detect_conflicts(forget_grad, retain_grad)
        
        # Apply the chosen strategy
        if self.surgery_strategy == "sago":
            return self._apply_sago(forget_grad, retain_grad, conflicts)
        elif self.surgery_strategy == "sago_prefer_retain":
            return self._apply_sago_prefer_retain(forget_grad, retain_grad, conflicts)
        elif self.surgery_strategy == "pcgrad":
            return self._apply_pcgrad(forget_grad, retain_grad)
        elif self.surgery_strategy == "sum":
            return self._apply_sum(forget_grad, retain_grad)
        else:
            raise ValueError(f"Unknown surgery strategy: {self.surgery_strategy}")

    @staticmethod
    def _dot_product(g1: List[torch.Tensor], g2: List[torch.Tensor]) -> torch.Tensor:
        """Calculate global dot product between two gradient lists."""
        dp = torch.tensor(0.0, device=g1[0].device)
        for t1, t2 in zip(g1, g2):
            dp += (t1 * t2).sum()
        return dp

    def _combine_grads(
        self,
        forget_grads: List[torch.Tensor],
        retain_grads: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Combine gradients using gradient surgery strategy.
        """
        if self.surgery_strategy == "pcgrad_global":
            return self._apply_pcgrad_global(forget_grads, retain_grads)
        elif self.surgery_strategy == "blur":
            return self._apply_blur(forget_grads, retain_grads)
        else:
            combined_grads = []
            for forget_grad, retain_grad in zip(forget_grads, retain_grads):
                combined_grad = self._apply_surgery_strategy(forget_grad, retain_grad)
                combined_grads.append(combined_grad)

            return combined_grads
    
    def _get_gradient_from_param(self, param: nn.Parameter) -> torch.Tensor:
        """Get gradient from parameter based on DeepSpeed configuration."""
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage in [1, 2]:
                return safe_get_full_grad(param)
            else:
                return safe_get_local_grad(param)
        else:
            return param.grad

    def _accumulate_grads_from_params(
        self,
        buffer: Optional[List[torch.Tensor]], 
        params: List[nn.Parameter]
    ) -> List[torch.Tensor]:
        """
        Accumulate gradients from parameters.
        """
        if buffer is None:
            return [self._get_gradient_from_param(p).clone().detach() for p in params]
        else:
            for buf_grad, param in zip(buffer, params):
                grad = self._get_gradient_from_param(param).detach()
                buf_grad.add_(grad)
            return buffer

    def _clear_grads(self, model: nn.Module, buffer: List[torch.Tensor], params: List[nn.Parameter]):
        if self.is_deepspeed_enabled:
            self.accelerator.deepspeed_engine_wrapped.engine.zero_grad()
            if self.accelerator.state.deepspeed_plugin.zero_stage in [1, 2]:
                model.zero_grad()
            else:
                for buf_grad, p in zip(buffer, params):
                    safe_set_local_grad(p, torch.zeros_like(buf_grad))
        else:
            model.zero_grad()

    def _empty_cache(self):
        if is_torch_xpu_available():
            torch.xpu.empty_cache()
        elif is_torch_mlu_available():
            torch.mlu.empty_cache()
        elif is_torch_musa_available():
            torch.musa.empty_cache()
        elif is_torch_npu_available():
            torch.npu.empty_cache()
        elif is_torch_mps_available(min_version="2.0"):
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()

    def _reset_accum_buffers(self, forget: bool = True, retain: bool = True):
        """
        Reset accumulation buffers to free memory.
        
        Args:
            forget: Whether to reset the forget gradient buffer
            retain: Whether to reset the retain gradient buffer
        """
        if forget:
            self._accum_forget = None
        if retain:
            self._accum_retain = None
        
        if forget or retain:
            self._empty_cache()
    
    def _sync_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Synchronize gradient tensors across distributed processes.
        
        Args:
            gradients: List of gradient tensors to synchronize
            
        Returns:
            List of synchronized gradient tensors
        """
        if self.is_deepspeed_enabled and self.accelerator.state.deepspeed_plugin.zero_stage in [2, 3]:
            return gradients
        
        # Use accelerator's reduce method to synchronize gradients across processes
        synced_grads = []
        for grad in gradients:
            synced_grad = self.accelerator.reduce(grad, reduction="mean")
            synced_grads.append(synced_grad)
        return synced_grads

    def _get_trainable_params(self, model: nn.Module) -> List[nn.Parameter]:
        """
        Get trainable parameters, handling DeepSpeed Stage 3 compatibility.
        In DeepSpeed Stage 3, all parameters have requires_grad=True, but frozen parameters
        have learning rate 0 in the optimizer parameter groups.
        """
        if self.is_deepspeed_enabled and self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.trainable_params_regex != [".*"]:
            trainable_params = []
            for name, param in model.named_parameters():
                if any(re.fullmatch("module." + pattern, name) for pattern in self.trainable_params_regex):
                    trainable_params.append(param)
            # print(f"Training step with {len(trainable_params)} trainable parameters.")
            return trainable_params
        else:
            return [p for p in model.parameters() if p.requires_grad]

    def _write_combined_grads_to_params(self, params: List[torch.Tensor], combined_grads: List[torch.Tensor]):
        """
        Write combined gradients back to parameters.
        """
        for param, cg in zip(params, combined_grads):
            if self.is_deepspeed_enabled:
                if self.accelerator.state.deepspeed_plugin.zero_stage in [1, 2]:
                    safe_set_full_grad(param, cg)
                else:
                    safe_set_local_grad(param, cg)
            else:
                param.grad = cg

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        kwargs = {}
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        # Parameters list (buffers will be created on first accumulation)
        params = self._get_trainable_params(model)
        
        # During accumulation window: collect gradients without sync
        # Use no_sync to prevent automatic gradient synchronization during accumulation
        # IMPORTANT: Both forward and backward passes must be inside no_sync context
        with self.accelerator.no_sync(model):
            with self.compute_loss_context_manager():
                forget_loss, retain_loss = self.compute_losses(model, inputs)

            if self.args.n_gpu > 1:
                forget_loss = forget_loss.mean()
                retain_loss = retain_loss.mean()

            # 1) forget_loss backward
            if self.is_deepspeed_enabled:
                if self.accelerator.sync_gradients:
                    self.accelerator.deepspeed_engine_wrapped.engine.set_gradient_accumulation_boundary(is_boundary=True)
                scaled_forget_loss = forget_loss / self.accelerator.gradient_accumulation_steps
                self.accelerator.deepspeed_engine_wrapped.engine.backward(scaled_forget_loss, scale_wrt_gas=False, **kwargs)
            else:
                self.accelerator.backward(forget_loss, **kwargs)
            self._accum_forget = self._accumulate_grads_from_params(self._accum_forget, params)
            self._clear_grads(model, self._accum_forget, params)

            # 2) retain_loss backward
            if self.is_deepspeed_enabled:
                if self.accelerator.sync_gradients:
                    self.accelerator.deepspeed_engine_wrapped.engine.set_gradient_accumulation_boundary(is_boundary=True)
                scaled_retain_loss = retain_loss / self.accelerator.gradient_accumulation_steps
                self.accelerator.deepspeed_engine_wrapped.engine.backward(scaled_retain_loss, scale_wrt_gas=False, **kwargs)
            else:
                self.accelerator.backward(retain_loss, **kwargs)
            self._accum_retain = self._accumulate_grads_from_params(self._accum_retain, params)
            self._clear_grads(model, self._accum_retain, params)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            self._empty_cache()
        
        # End of accumulation window: perform gradient surgery
        if self.accelerator.sync_gradients:
            # We have accumulated gradients, sync them directly and then combine
            
            # 1) Sync forget gradients
            forget_grads = self._sync_gradients(self._accum_forget)
            self._reset_accum_buffers(forget=True, retain=False)

            # 2) Sync retain gradients
            retain_grads = self._sync_gradients(self._accum_retain)
            self._reset_accum_buffers(forget=False, retain=True)

            # 3) Perform gradient surgery on synchronized gradients
            combined_grads = self._combine_grads(forget_grads, retain_grads)
            
            # 4) Write combined grads back to parameters
            del forget_grads, retain_grads
            self._write_combined_grads_to_params(params, combined_grads)

            if self.is_deepspeed_enabled:
                self.accelerator.deepspeed_engine_wrapped.engine.step()
            
            self._empty_cache()

        self.accelerator.unwrap_model(model).report_metrics(forget_loss=-forget_loss.detach(), retain_loss=retain_loss.detach())

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return loss.detach() / self.args.gradient_accumulation_steps


class GradDiffWithSurgery(GradSurgeryMixin, GradDiff):
    """
    GradDiff trainer with gradient surgery capabilities.
    
    Combines the basic GradDiff approach with advanced gradient surgery techniques
    to handle conflicts between forget and retain gradients.
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.0,
        retain_loss_type: str = "NLL",  # ["NLL", "KL"]
        surgery_strategy: str = "sago",  # ["sago", "sago_prefer_retain", "pcgrad", "pcgrad_global", "blur", "sum"]
        *args,
        **kwargs,
    ):
        # Initialize GradDiff first
        GradDiff.__init__(self, gamma=gamma, alpha=alpha, retain_loss_type=retain_loss_type, *args, **kwargs)
        
        # Initialize gradient surgery mixin
        self.__init_grad_surgery__(
            surgery_strategy=surgery_strategy,
        )

    def compute_losses(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GradDiff-specific: forget (negative NLL) and retain (NLL or KL)
        """
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        
        return forget_loss, retain_loss


class SimNPOWithSurgery(GradSurgeryMixin, SimNPO):
    """
    SimNPO trainer with gradient surgery capabilities.
    
    Combines the SimNPO (Simplified Negative Preference Optimization) approach with advanced 
    gradient surgery techniques to handle conflicts between forget and retain gradients.
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.0,
        retain_loss_type: str = "NLL",  # ["NLL", "KL"]
        delta: float = 0.0,
        beta: float = 1.0,
        surgery_strategy: str = "sago",  # ["sago", "sago_prefer_retain", "pcgrad", "pcgrad_global", "blur", "sum"]
        *args,
        **kwargs,
    ):
        # Initialize SimNPO first
        SimNPO.__init__(self, gamma=gamma, alpha=alpha, retain_loss_type=retain_loss_type, delta=delta, beta=beta, *args, **kwargs)
        
        # Initialize gradient surgery mixin
        self.__init_grad_surgery__(
            surgery_strategy=surgery_strategy,
        )
    
    def compute_losses(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SimNPO-specific loss computation: forget loss uses SimNPO formula, retain loss uses NLL or KL
        """
        from trainer.utils import compute_batch_nll
        
        forget_inputs = inputs["forget"]
        forget_labels = forget_inputs["labels"]
        loss_mask = forget_labels != -100
        forget_loss, _ = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        
        return forget_loss, retain_loss


class NPOWithSurgery(GradSurgeryMixin, NPO):
    """
    NPO trainer with gradient surgery capabilities.
    
    Combines the NPO (Negative Preference Optimization) approach with advanced 
    gradient surgery techniques to handle conflicts between forget and retain gradients.
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.0,
        retain_loss_type: str = "NLL",  # ["NLL", "KL"]
        beta: float = 1.0,
        surgery_strategy: str = "sago",  # ["sago", "sago_prefer_retain", "pcgrad", "pcgrad_global", "blur", "sum"]
        *args,
        **kwargs,
    ):
        # Initialize NPO first  
        NPO.__init__(self, gamma=gamma, alpha=alpha, retain_loss_type=retain_loss_type, beta=beta, *args, **kwargs)
        
        # Initialize gradient surgery mixin
        self.__init_grad_surgery__(
            surgery_strategy=surgery_strategy,
        )
    
    def compute_losses(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NPO-specific loss computation: forget loss uses DPO, retain loss uses NLL or KL
        """
        from trainer.utils import compute_dpo_loss
        
        forget_inputs = inputs["forget"]
        forget_loss, _ = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        
        return forget_loss, retain_loss


class RMUWithSurgery(GradSurgeryMixin, RMU):
    """
    RMU trainer with gradient surgery capabilities.
    
    Combines the RMU (Representation Misdirection for Unlearning) approach with advanced 
    gradient surgery techniques to handle conflicts between forget and retain gradients.
    
    RMU uses activation-based loss to steer model representations away from the forget
    data while preserving performance on retain data. This surgery variant applies
    gradient surgery to resolve conflicts between the activation-based forget loss
    and the retain loss.
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.0,
        retain_loss_type: str = "NLL",  # ["NLL", "KL", "EMBED_DIFF"]
        module_regex: str = "model\\.layers\\.7",
        trainable_params_regex: List[str] = ["model\\.layers\\.(5|6|7)\\.mlp\\.down_proj\\.weight"],
        steering_coeff: float = 20,
        surgery_strategy: str = "sago",  # ["sago", "sago_prefer_retain", "pcgrad", "pcgrad_global", "blur", "sum"]
        *args,
        **kwargs,
    ):
        # Initialize RMU first
        RMU.__init__(
            self,
            gamma=gamma,
            alpha=alpha,
            retain_loss_type=retain_loss_type,
            module_regex=module_regex,
            trainable_params_regex=trainable_params_regex,
            steering_coeff=steering_coeff,
            *args,
            **kwargs,
        )
        
        # Initialize gradient surgery mixin
        self.__init_grad_surgery__(
            surgery_strategy=surgery_strategy,
        )
    
    def compute_losses(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RMU-specific loss computation: 
        - forget loss uses activation misdirection (steer activations to random control vector)
        - retain loss uses NLL, KL, or EMBED_DIFF (activation matching with reference model)
        """
        # Compute forget loss using activation misdirection
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        model_forget_activations, forget_outputs = self.forward_with_cache(
            model, forget_inputs, self.model_module, no_grad=False
        )
        # If multiple datasets or concepts need unlearning, pass the control vector during processing; 
        # otherwise, default to a random vector during training.
        control_vec = forget_inputs.get(
            "control_vec", self.get_control_vector(model_forget_activations.shape[-1])
        )
        control_vec = control_vec.to(
            dtype=model_forget_activations.dtype, device=model_forget_activations.device
        )
        control_vec = control_vec.expand_as(model_forget_activations)
        mask = forget_inputs["labels"] != -100  # Shape: [b, s]
        forget_loss = self.compute_activation_loss(
            model_forget_activations, control_vec, mask
        )

        # Compute retain loss
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        
        return forget_loss, retain_loss
