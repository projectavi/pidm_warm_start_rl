# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import subprocess
from pathlib import Path

from pidm_imitation.utils.logger import Logger

log = Logger.get_logger(__name__)


def _run_git_command(
    args_list: list[str], cwd: str = None
) -> subprocess.CompletedProcess:
    if cwd:
        return subprocess.run(
            args_list, capture_output=True, text=True, encoding="utf-8", cwd=cwd
        )
    else:
        return subprocess.run(
            args_list, capture_output=True, text=True, encoding="utf-8"
        )


def is_git_repo(cwd: str = None) -> bool:
    if not Path(cwd).is_dir():
        log.warning(f"Unable to find directory {cwd}")
        return False
    args_list = ["git", "rev-parse", "--is-inside-work-tree"]
    result = _run_git_command(args_list, cwd=cwd)
    return result.returncode == 0 and result.stdout.strip() == "true"


def get_head_commitid(short: bool = False, cwd: str = None) -> str | None:
    if short:
        args_list = ["git", "rev-parse", "--short", "HEAD"]
    else:
        args_list = ["git", "rev-parse", "HEAD"]

    result = _run_git_command(args_list, cwd=cwd)

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        log.warning("Could not get git hash")
        return None


def get_branch_name(cwd: str = None) -> str | None:
    args_list = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    result = _run_git_command(args_list, cwd=cwd)

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        log.warning("Could not get the current branch name")
        return None
