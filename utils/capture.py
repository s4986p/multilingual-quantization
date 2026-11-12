import torch
from torch import Tensor
from torch.nn import Linear, Module
from transformers import PreTrainedModel


class StopForward(Exception):
    pass


def capture_layer_inputs(
    model: PreTrainedModel,
    layer: Module,
    samples: list[Tensor],
) -> list[tuple[tuple, dict]]:
    layer_inputs = []

    def hook(
        module: Module,
        args: tuple,
        kwargs: dict,
    ) -> None:
        layer_inputs.append((args, kwargs))
        raise StopForward

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    for input_ids in samples:
        try:
            model(input_ids[None, :].to(model.device), use_cache=False)
        except StopForward:
            pass
    handle.remove()
    return layer_inputs


def compute_abs_mean_and_hessian(
    layer: Module,
    linear: Linear,
    layer_inputs: list[tuple[tuple, dict]],
) -> tuple[Tensor, Tensor]:
    abs_sum = torch.zeros(linear.in_features, device=linear.weight.device)
    hessian = torch.zeros(linear.in_features, linear.in_features, device=linear.weight.device)
    num_tokens = 0

    def hook(
        module: Module,
        args: tuple[Tensor, ...],
    ) -> None:
        nonlocal num_tokens
        activations = args[0].reshape(-1, args[0].shape[-1]).float()
        abs_sum.add_(activations.abs().sum(dim=0))
        hessian.add_(activations.T @ activations)
        num_tokens += activations.shape[0]

    handle = linear.register_forward_pre_hook(hook)
    for layer_args, layer_kwargs in layer_inputs:
        layer(*layer_args, **layer_kwargs)
    handle.remove()
    return abs_sum / num_tokens, hessian
