"""Push cleaned datasets to HuggingFace AksaraLLM org.

Datasets created:
  - AksaraLLM/aksara-pretrain-clean-v1
  - AksaraLLM/aksara-sft-clean-v1
  - AksaraLLM/aksara-bahasa-daerah-v1

Reads HF_TOKEN from env.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi(token=os.environ["HF_TOKEN"])
ORG = "AksaraLLM"

def push(repo_id, local_dir, card_path, repo_type="dataset"):
    print(f"[push] {repo_id} ←  {local_dir}")
    create_repo(repo_id, repo_type=repo_type, exist_ok=True, token=api.token)
    # Upload card as README.md at repo root
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Add dataset card",
    )
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=["*.md", ".DS_Store"],
        commit_message="Initial upload of clean data",
    )
    print(f"  done → https://huggingface.co/datasets/{repo_id}")


base = Path("out/final")

push(f"{ORG}/aksara-pretrain-clean-v1", base / "pretrain", base / "pretrain_card.md")
push(f"{ORG}/aksara-sft-clean-v1", base / "sft", base / "sft_card.md")
push(f"{ORG}/aksara-bahasa-daerah-v1", base / "bahasa-daerah", base / "bahasa_daerah_card.md")
