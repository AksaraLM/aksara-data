"""Push cleaned datasets to HuggingFace AksaraLLM org.

Datasets created:
  - AksaraLLM/aksara-pretrain-clean-v1
  - AksaraLLM/aksara-sft-clean-v1
  - AksaraLLM/aksara-bahasa-daerah-v1

Reads HF_TOKEN from env. Card markdown templates live in pretrain/cards/.
Missing card files are tolerated (a warning is printed and upload continues
without the README).
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi(token=os.environ["HF_TOKEN"])
ORG = "AksaraLLM"

# Resolve cards relative to this script, not the caller's cwd.
CARDS_DIR = Path(__file__).parent / "cards"


def push(repo_id, local_dir, card_path, repo_type="dataset"):
    print(f"[push] {repo_id} \u2190  {local_dir}")
    create_repo(repo_id, repo_type=repo_type, exist_ok=True, token=api.token)

    if card_path and Path(card_path).exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Add dataset card",
        )
    else:
        print(f"  [warn] card not found at {card_path}; skipping README upload")

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=["*.md", ".DS_Store"],
        commit_message="Initial upload of clean data",
    )
    print(f"  done \u2192 https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    base = Path("out/final")
    push(f"{ORG}/aksara-pretrain-clean-v1", base / "pretrain", CARDS_DIR / "pretrain_card.md")
    push(f"{ORG}/aksara-sft-clean-v1", base / "sft", CARDS_DIR / "sft_card.md")
    push(f"{ORG}/aksara-bahasa-daerah-v1", base / "bahasa-daerah", CARDS_DIR / "bahasa_daerah_card.md")
