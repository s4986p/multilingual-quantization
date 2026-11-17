import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import PreTrainedModel

from utils.paths import make_result_name

REPO_NAME = "s4986p/multilingual-quantization"


def save_model(
    hub_path: str,
    model: PreTrainedModel,
) -> None:
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        HfApi().upload_folder(folder_path=directory, path_in_repo=hub_path, repo_id=REPO_NAME, commit_message=f"save {hub_path}")
    print(f"saved {hub_path}")


def download_model(
    hub_path: str,
) -> str:
    return str(Path(snapshot_download(REPO_NAME, allow_patterns=[f"{hub_path}/*.json", f"{hub_path}/model*.safetensors"])) / hub_path)


def result_exists(
    experiment_name: str,
    hub_path: str,
    file_name: str,
) -> bool:
    return HfApi().file_exists(REPO_NAME, f"{experiment_name}/{hub_path}/{file_name}.json")


def save_result(
    experiment_name: str,
    hub_path: str,
    file_name: str,
    result: dict,
) -> None:
    path = f"{experiment_name}/{hub_path}/{file_name}.json"
    HfApi().upload_file(path_or_fileobj=json.dumps(result, indent=2, ensure_ascii=False).encode(), path_in_repo=path, repo_id=REPO_NAME, commit_message=f"save {path}")
    print(f"saved {path}")


def load_result(
    experiment_name: str,
    hub_path: str,
    file_name: str,
) -> dict:
    return json.loads(Path(hf_hub_download(REPO_NAME, f"{experiment_name}/{hub_path}/{file_name}.json")).read_text())


def load_results(
    experiment_name: str,
    hub_path: str,
    max_generation_len: int,
    seeds: tuple[int, ...],
) -> list[dict]:
    return [load_result(experiment_name, hub_path, make_result_name(max_generation_len, seed)) for seed in seeds]
