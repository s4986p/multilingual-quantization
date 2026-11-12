import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import PreTrainedModel

REPO_NAME = "s4986p/multilingual-quantization"


def upload_file(
    path: str,
    data: bytes,
) -> None:
    HfApi().upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=REPO_NAME, commit_message=f"save {path}")
    print(f"saved {path}")


def download_file(
    path: str,
) -> str:
    return hf_hub_download(REPO_NAME, path)


def save_model(
    path: str,
    model: PreTrainedModel,
) -> None:
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        HfApi().upload_folder(folder_path=directory, path_in_repo=path, repo_id=REPO_NAME, commit_message=f"save {path}")
    print(f"saved {path}")


def download_model(
    path: str,
) -> str:
    return str(Path(snapshot_download(REPO_NAME, allow_patterns=[f"{path}/*.json", f"{path}/model*.safetensors"])) / path)


def result_exists(
    script_name: str,
    target: str,
    file_name: str,
) -> bool:
    return HfApi().file_exists(REPO_NAME, f"{script_name}/{target}/{file_name}.json")


def save_result(
    script_name: str,
    target: str,
    file_name: str,
    result: dict,
) -> None:
    upload_file(f"{script_name}/{target}/{file_name}.json", json.dumps(result, indent=2, ensure_ascii=False).encode())


def load_result(
    script_name: str,
    target: str,
    file_name: str,
) -> dict:
    return json.loads(Path(download_file(f"{script_name}/{target}/{file_name}.json")).read_text())
