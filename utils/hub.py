from tempfile import TemporaryDirectory

from huggingface_hub import HfApi
from transformers import PreTrainedModel

REPO_NAME = "s4986p/multilingual-quantization"


def save_model(
    path: str,
    model: PreTrainedModel,
) -> None:
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        HfApi().upload_folder(folder_path=directory, path_in_repo=path, repo_id=REPO_NAME, commit_message=f"save {path}")
    print(f"saved {path}")
