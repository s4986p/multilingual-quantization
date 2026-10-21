from dataclasses import dataclass
from pathlib import Path

from simple_parsing import parse

from primitives.plot import load_results, plot_benchmark, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.5-2B"


def main() -> None:
    config = parse(Config)
    model_short_name = config.model_name.split("/")[-1]
    results = load_results("mgsm", config.model_name, "seed=*", ["bf16", "gptq", "awq"])
    fig = plot_benchmark(results, results[f"bf16/{config.model_name}"], f"{model_short_name} mgsm")
    save_figure(fig, Path(f"figures/{model_short_name}/mgsm.png"))


if __name__ == "__main__":
    main()
