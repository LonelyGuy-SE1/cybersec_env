"""Replicate ``openenv push`` while bypassing the rate-limited ``whoami()``.

The OpenEnv push command short-circuits on ``whoami()`` for an auth check, but
HF rate-limits ``/whoami-v2`` aggressively. The token in
``~/.cache/huggingface`` is still valid; only the auth probe fails. We reuse
OpenEnv's staging logic verbatim and call ``HfApi`` directly.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from openenv.cli.commands.push import (
    DEFAULT_PUSH_IGNORE_PATTERNS,
    _prepare_staging_directory,
)


def main() -> int:
    repo_id = "Lonelyguyse1/cybersec"
    env_dir = Path(__file__).resolve().parent.parent / "cybersec"
    env_name = "cybersec"

    # OpenEnv default ignores + a few project-specific ones we want to keep
    # out of the deployed Space.
    ignore_patterns = list(DEFAULT_PUSH_IGNORE_PATTERNS) + [
        "outputs/*",
        "outputs/",
        "tests/*",
        "tests/",
        "docs/*",
        "docs/",
        "*.pyc",
    ]

    print(f"[push] env_dir = {env_dir}")
    print(f"[push] ignore_patterns = {ignore_patterns}")

    with tempfile.TemporaryDirectory(prefix="cybersec-stage-") as staging_root:
        staging_dir = Path(staging_root) / env_name
        _prepare_staging_directory(
            env_dir=env_dir,
            env_name=env_name,
            staging_dir=staging_dir,
            ignore_patterns=ignore_patterns,
            base_image=None,
            enable_interface=True,
        )

        print(f"[push] staged in {staging_dir}; uploading to {repo_id} ...")
        api = HfApi()
        # ``delete_patterns=['*']`` makes upload_folder commit a wholesale
        # delete of any file that isn't in the staging tree, atomically with
        # the upload. This wipes stale modules (e.g. an older
        # ``server/attacker_policy.py``) that ``upload_folder`` would
        # otherwise leave behind and that break imports at boot.
        url = api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            delete_patterns=["*"],
            commit_message=(
                "fix(server): clean tree + top-level import fallback for HF runtime"
            ),
        )
        print(f"[push] done: {url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
