from tempfile import TemporaryDirectory

from huggingface_hub import HfApi
from transformers import PreTrainedModel


def get_repo_name() -> str:
    return f"{HfApi().whoami()['name']}/multilingual-quantization"


def save_model(
    path: str,
    model: PreTrainedModel,
) -> None:
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        HfApi().upload_folder(folder_path=directory, path_in_repo=path, repo_id=get_repo_name(), commit_message=f"save {path}")
    print(f"saved {path}")
