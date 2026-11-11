from dataclasses import dataclass
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module
from transformers import AutoTokenizer, PreTrainedModel, set_seed

from utils.hub import save_model
from utils.model import load_model
from utils.paths import make_qat_path
from utils.quantization import MXFP4_GROUP_SIZE, find_fp4_scale, round_to_fp4
from utils.training import load_samples, train


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    # data to train on
    think_language: Literal["en", "native"] = "en"
    seed: int = 0
    max_generation_len: int = 16384


class STELinear(Module):
    def __init__(
        self,
        linear: Linear,
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.scale = find_fp4_scale(self.groups())[..., None]

    def groups(
        self,
    ) -> Tensor:
        return self.weight.reshape(self.weight.shape[0], -1, MXFP4_GROUP_SIZE)

    def quantize(
        self,
    ) -> Tensor:
        # detach only rounding, because gradient of quantized weight must go to weight
        with torch.no_grad():
            quantized = round_to_fp4(self.groups(), self.scale).reshape(self.weight.shape)
        return self.weight + (quantized - self.weight).detach()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return torch.nn.functional.linear(x, self.quantize(), self.bias)


def replace_linears(
    model: PreTrainedModel,
) -> list[tuple[str, Linear, STELinear]]:
    layers = model.model.language_model.layers
    linears = [(name, linear) for name, linear in layers.named_modules() if isinstance(linear, Linear)]
    replaced = []
    for name, linear in linears:
        ste_linear = STELinear(linear)
        parent_name, child_name = name.rsplit(".", 1)
        setattr(layers.get_submodule(parent_name), child_name, ste_linear)
        replaced.append((name, linear, ste_linear))
    return replaced


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()

    samples = load_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.think_language)

    # 1) use ste linears, because gradient must go through mxfp4 rounding
    replaced = replace_linears(model)
    train(model, [ste_linear.weight for _, _, ste_linear in replaced], samples, config.lr, config.num_epochs, config.warmup_ratio, config.batch_size)

    # 2) put quantized weights back into linears and save
    layers = model.model.language_model.layers
    with torch.no_grad():
        for name, linear, ste_linear in replaced:
            linear.weight.copy_(ste_linear.quantize())
            parent_name, child_name = name.rsplit(".", 1)
            setattr(layers.get_submodule(parent_name), child_name, linear)
    model.to(torch.bfloat16)
    save_model(make_qat_path(config.model_name, config.think_language), model)


if __name__ == "__main__":
    main()
