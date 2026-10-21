from functools import partial

import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList
from transformers import PreTrainedModel


class StopForward(Exception):
    pass


def capture_layer_inputs(
    model: PreTrainedModel,
    layer: Module,
    samples: list[Tensor],
) -> list[tuple[tuple, dict]]:
    inputs = []

    def hook(
        module: Module,
        args: tuple,
        kwargs: dict,
    ) -> None:
        inputs.append((args, kwargs))
        raise StopForward

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    for input_ids in samples:
        try:
            model(input_ids.unsqueeze(0).to(model.device), use_cache=False)
        except StopForward:
            pass
    handle.remove()
    return inputs


def get_layers(
    model: PreTrainedModel,
) -> ModuleList:
    return model.get_submodule("model.language_model.layers")


def get_linears(
    module: Module,
) -> dict[str, Linear]:
    return {name: linear for name, linear in module.named_modules() if isinstance(linear, Linear) and not name.startswith("per_layer_")}


@torch.no_grad()
def find_linear_groups(
    layer: Module,
    args: tuple,
    kwargs: dict,
) -> list[dict[str, Linear]]:
    linear_groups = []

    def hook(
        name: str,
        linear: Linear,
        hook_args: tuple[Tensor, ...],
    ) -> None:
        for x, linear_group in linear_groups:
            if x is hook_args[0]:
                linear_group[name] = linear
                return
        linear_groups.append((hook_args[0], {name: linear}))

    handles = [linear.register_forward_pre_hook(partial(hook, name)) for name, linear in get_linears(layer).items()]
    layer(*args, **kwargs)
    for handle in handles:
        handle.remove()
    return [linear_group for _, linear_group in linear_groups]
