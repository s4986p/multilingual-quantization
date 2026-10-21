import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import save_file
from torch import Tensor
from transformers import PreTrainedModel


def get_repo_name() -> str:
    return f"{HfApi().whoami()['name']}/multilingual-quantization"


def make_gptq_path(
    model_name: str,
    languages: list[str],
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    return f"gptq/{model_name}/languages={'_'.join(languages)},num_bits={num_bits},group_size={group_size},num_samples={num_samples}"


def make_awq_path(
    model_name: str,
    languages: list[str],
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    return f"awq/{model_name}/languages={'_'.join(languages)},num_bits={num_bits},group_size={group_size},num_samples={num_samples}"


def save_quantized(
    path: str,
    model: PreTrainedModel,
    grid: dict[str, Tensor],
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        save_file({name: tensor.cpu().contiguous() for name, tensor in grid.items()}, f"{directory}/grid.safetensors")
        api.upload_folder(folder_path=directory, path_in_repo=path, repo_id=repo_name, commit_message=f"Save {path}")
    print(f"saved {path}")


def download_model(
    path: str,
) -> str:
    return str(Path(snapshot_download(get_repo_name(), allow_patterns=[f"{path}/*"], ignore_patterns=[f"{path}/grid.safetensors"])) / path)


def get_pending_targets(
    entry_name: str,
    model_name: str,
    file_name: str,
) -> list[str]:
    files = set(HfApi().list_repo_files(get_repo_name()))
    targets = [f"bf16/{model_name}"]
    for file in sorted(files):
        if file.startswith((f"gptq/{model_name}/", f"awq/{model_name}/")) and file.endswith("/grid.safetensors"):
            targets.append(str(Path(file).parent))
    targets = [target for target in targets if f"{entry_name}/{target}/{file_name}.json" not in files]
    print(f"{entry_name} pending {len(targets)}", *targets, sep="\n  ")
    return targets


def save_result(
    path: str,
    file_name: str,
    result: dict,
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=json.dumps(result, indent=2, ensure_ascii=False).encode(), path_in_repo=f"{path}/{file_name}.json", repo_id=repo_name, commit_message=f"Save {path}/{file_name}.json")
    print(f"saved {path}/{file_name}.json")
