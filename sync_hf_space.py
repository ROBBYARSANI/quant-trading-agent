"""
Hugging Face Static Space Dashboard Synchronizer.
Pushes index.html and latest_telemetry.json directly to Hugging Face Static Space
using the huggingface_hub API (no paid Docker tier required).
"""

import os
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HF_SPACE_REPO = os.getenv("HF_SPACE_REPO", "PeopleNw/quant-trading-agent")
BASE_DIR = Path(__file__).resolve().parent


def sync_dashboard_to_hf_space(repo_id: str = HF_SPACE_REPO) -> bool:
    token = os.getenv("HF_TOKEN")
    if not token or not token.strip():
        print(f"[INFO] HF_TOKEN environment variable is not configured. Skipping Hugging Face Space sync for '{repo_id}'.")
        return False

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[WARNING] huggingface_hub library is not installed. Please install 'huggingface-hub'.")
        return False

    api = HfApi(token=token.strip())
    print(f"[INFO] Initiating Static Space synchronization to Hugging Face: {repo_id}...")

    # 1. Create/Ensure Static Space README metadata
    space_readme_content = """---
title: Autonomous Quant Trading Agent
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# Autonomous Quant Trading Agent — Live Dashboard
Automated 24/7 Cryptocurrency Scanner & Paper Trading Engine powered by XGBoost Triple-Barrier, Half-Kelly Risk Allocation, and FinBERT Sentiment Filter.
Hosted on Hugging Face Free Static Space.
"""
    temp_readme = BASE_DIR / "models" / "space_README.md"
    temp_readme.parent.mkdir(parents=True, exist_ok=True)
    temp_readme.write_text(space_readme_content, encoding="utf-8")

    try:
        api.upload_file(
            path_or_fileobj=str(temp_readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
            commit_message="Configure Free Static Space Metadata [skip ci]",
        )
        print("  ✓ README.md metadata verified.")
    except Exception as exc:
        print(f"  ℹ Note on space README check: {exc}")

    # 2. Upload index.html and latest_telemetry.json
    files_to_sync = [
        ("index.html", "index.html"),
        ("latest_telemetry.json", "latest_telemetry.json"),
    ]

    synced_count = 0
    for local_name, remote_name in files_to_sync:
        local_path = BASE_DIR / local_name
        # Fallback to models/ if not in root
        if not local_path.exists():
            if local_name == "index.html":
                local_path = BASE_DIR / "models" / "dashboard.html"
            elif local_name == "latest_telemetry.json":
                local_path = BASE_DIR / "models" / "latest_telemetry.json"

        if not local_path.exists():
            print(f"  ⚠ File not found: {local_path}, skipping.")
            continue

        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="space",
                commit_message=f"Auto-sync live telemetry: {remote_name}",
            )
            print(f"  ✓ Uploaded {remote_name} -> {repo_id}")
            synced_count += 1
        except Exception as exc:
            print(f"  ✗ Failed to upload {remote_name} to {repo_id}: {exc}")

    if synced_count > 0:
        print(f"[SUCCESS] Successfully synced {synced_count} files to https://huggingface.co/spaces/{repo_id}")
        return True
    return False


if __name__ == "__main__":
    success = sync_dashboard_to_hf_space()
    sys.exit(0)
