"""Hugging Face Hub access: safe reference normalization, revision pinning, verified downloads.

``normalize_model_ref`` is pure and used by the CLI/config layer (also in --dry-run).
``HubClient`` lazily imports ``huggingface_hub`` so that importing this module never needs
network access or third-party packages.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..errors import ExitCode, PipelineError
from ..fsutil import git_blob_sha1, sha256_file

_SEGMENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
_REPO_ID_RE = re.compile(rf"^(?:{_SEGMENT}/)?{_SEGMENT}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HF_HOSTS = {"huggingface.co", "www.huggingface.co"}


def _invalid(message: str, hint: str | None = None) -> PipelineError:
    return PipelineError(ExitCode.CONFIG, "E_MODEL_REF_INVALID", message, hint=hint)


def validate_repo_id(repo_id: str) -> str:
    if "--" in repo_id or ".." in repo_id or not _REPO_ID_RE.match(repo_id):
        raise _invalid(
            f"Not a valid Hugging Face repository ID: {repo_id!r}",
            hint="Use 'org/name' (letters, digits, '.', '_', '-'; no '--' or '..').",
        )
    return repo_id


def validate_revision(revision: str) -> str:
    if ".." in revision or not _REVISION_RE.match(revision):
        raise _invalid(f"Not a valid revision: {revision!r}", hint="Use a commit SHA, tag or branch name.")
    return revision


def normalize_model_ref(text: str) -> str:
    """Accept a repo ID or canonical model-root URL; return the repo ID. Never shell-interpreted."""
    if not isinstance(text, str):
        raise _invalid("Model reference must be a string")
    value = text.strip()
    if not value or len(value) > 300 or any(ord(c) < 32 or c.isspace() for c in value):
        raise _invalid(f"Model reference contains whitespace/control characters or is empty: {text!r}")
    if "://" not in value:
        if value.startswith(("/", ".", "~")) or "\\" in value:
            raise _invalid(
                f"{value!r} looks like a filesystem path, not a Hugging Face repo ID",
                hint="Use --model-path for local directories.",
            )
        return validate_repo_id(value.rstrip("/"))
    parts = urlsplit(value)
    if parts.scheme != "https":
        raise _invalid(f"Only https://huggingface.co URLs are accepted, got scheme {parts.scheme!r}")
    if parts.username or parts.password or parts.port or (parts.hostname or "").lower() not in _HF_HOSTS:
        raise _invalid(
            f"Only canonical https://huggingface.co/<org>/<name> URLs are accepted: {value!r}",
            hint="Arbitrary URLs and mirrors are rejected in v1.",
        )
    if parts.query or parts.fragment:
        raise _invalid(f"Model URL must not contain a query or fragment: {value!r}")
    segments = [s for s in parts.path.split("/") if s]
    if segments and segments[0] in {"datasets", "spaces", "api", "models"}:
        raise _invalid(f"URL does not point at a model repository root: {value!r}")
    if len(segments) > 2:
        raise _invalid(
            f"URL is not a model root (extra path {'/'.join(segments[2:])!r}): {value!r}",
            hint="Pass the revision separately with --revision instead of /tree/<rev> URLs.",
        )
    if not segments:
        raise _invalid(f"URL has no repository path: {value!r}")
    return validate_repo_id("/".join(segments))


def default_family(repo_id: str | None, local_path: str | None) -> str:
    if repo_id:
        return repo_id.split("/")[-1].lower()
    if local_path:
        return Path(local_path).name.lower() or "local-model"
    return "unknown"


@dataclass
class RepoFile:
    path: str
    size: int | None
    sha256: str | None  # LFS sha256 when available
    blob_id: str | None  # git blob sha1 for non-LFS files

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256, "blob_id": self.blob_id}


@dataclass
class RepoInfo:
    repo_id: str
    repo_type: str
    sha: str
    requested_revision: str | None
    files: list[RepoFile] = field(default_factory=list)
    gated: Any = None
    card_data: dict[str, Any] | None = None
    config: dict[str, Any] | None = None

    @property
    def total_size(self) -> int | None:
        sizes = [f.size for f in self.files]
        return None if any(s is None for s in sizes) else sum(sizes)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "sha": self.sha,
            "requested_revision": self.requested_revision,
            "files": [f.to_dict() for f in self.files],
            "gated": self.gated,
            "total_size": self.total_size,
        }


class HubClient:
    """Thin wrapper over huggingface_hub; the only place that touches the Hub API."""

    def __init__(self, cache_dir: str | os.PathLike[str], token: str | None) -> None:
        self.cache_dir = str(cache_dir)
        self.token = token

    def _hub(self):
        try:
            import huggingface_hub  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise PipelineError(
                ExitCode.ENVIRONMENT,
                "E_CONTROL_ENV_INCOMPLETE",
                "huggingface_hub is not installed in the control environment",
                hint="Run ./main.sh once without --dry-run to bootstrap .venv, or 'uv sync --frozen'.",
            ) from exc
        return huggingface_hub

    def _translate(self, exc: Exception, repo_id: str) -> PipelineError:
        name = type(exc).__name__
        if name == "GatedRepoError":
            return PipelineError(
                ExitCode.ACQUIRE,
                "E_HF_GATED",
                f"{repo_id} is gated and the token cannot access it",
                hint="Accept the license on the model page and export the token in the env var named by model.hf_token_env.",
            )
        if name in {"RepositoryNotFoundError", "RevisionNotFoundError"}:
            return PipelineError(
                ExitCode.ACQUIRE,
                "E_HF_NOT_FOUND",
                f"{repo_id}: {name} (private repos need a token; check spelling and revision)",
            )
        return PipelineError(ExitCode.ACQUIRE, "E_HF_API", f"Hugging Face API error for {repo_id}: {name}: {exc}")

    def resolve(self, repo_id: str, revision: str | None, repo_type: str = "model") -> RepoInfo:
        hub = self._hub()
        api = hub.HfApi(token=self.token)
        try:
            if repo_type == "dataset":
                info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
            else:
                info = api.model_info(repo_id, revision=revision, files_metadata=True)
        except Exception as exc:  # noqa: BLE001 - translated to a stable error
            raise self._translate(exc, repo_id) from exc
        files = []
        for sibling in info.siblings or []:
            lfs = getattr(sibling, "lfs", None)
            lfs_sha = None
            if lfs is not None:
                lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            files.append(
                RepoFile(
                    path=sibling.rfilename,
                    size=getattr(sibling, "size", None),
                    sha256=lfs_sha,
                    blob_id=getattr(sibling, "blob_id", None),
                )
            )
        card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
        card_dict = card.to_dict() if hasattr(card, "to_dict") else (card if isinstance(card, dict) else None)
        return RepoInfo(
            repo_id=repo_id,
            repo_type=repo_type,
            sha=info.sha,
            requested_revision=revision,
            files=sorted(files, key=lambda f: f.path),
            gated=getattr(info, "gated", None),
            card_data=card_dict,
            config=getattr(info, "config", None),
        )

    def snapshot(self, repo_id: str, sha: str, repo_type: str = "model", allow_patterns: list[str] | None = None) -> Path:
        hub = self._hub()
        try:
            path = hub.snapshot_download(
                repo_id,
                revision=sha,
                repo_type=repo_type,
                cache_dir=self.cache_dir,
                token=self.token,
                allow_patterns=allow_patterns,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, repo_id) from exc
        return Path(path)

    def file_exists(self, repo_id: str, filename: str, sha: str, repo_type: str = "model") -> bool:
        hub = self._hub()
        try:
            return bool(hub.file_exists(repo_id, filename, revision=sha, repo_type=repo_type, token=self.token))
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, repo_id) from exc

    def download_file(self, repo_id: str, filename: str, sha: str, repo_type: str = "model") -> Path:
        hub = self._hub()
        try:
            return Path(hub.hf_hub_download(repo_id, filename, revision=sha, repo_type=repo_type, cache_dir=self.cache_dir, token=self.token))
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, repo_id) from exc


def verify_snapshot(
    snapshot_dir: Path,
    info: RepoInfo,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Verify every file against Hub metadata. Returns {relative_path: sha256}.

    LFS files are checked against their published sha256, regular files against their git
    blob id. A mismatch or a missing file raises E_DOWNLOAD_CORRUPT.
    """
    hashes: dict[str, str] = {}
    problems = []
    for repo_file in info.files:
        local = snapshot_dir / repo_file.path
        if not local.exists():
            problems.append(f"missing {repo_file.path}")
            continue
        if repo_file.size is not None and local.stat().st_size != repo_file.size:
            problems.append(f"size mismatch {repo_file.path}: {local.stat().st_size} != {repo_file.size}")
            continue
        if progress:
            progress(repo_file.path)
        digest = sha256_file(local)
        hashes[repo_file.path] = digest
        if repo_file.sha256:
            if digest != repo_file.sha256.lower():
                problems.append(f"sha256 mismatch {repo_file.path}")
        elif repo_file.blob_id:
            if git_blob_sha1(local) != repo_file.blob_id:
                problems.append(f"git blob mismatch {repo_file.path}")
        else:
            problems.append(f"no published hash for {repo_file.path}")
    if problems:
        raise PipelineError(
            ExitCode.ACQUIRE,
            "E_DOWNLOAD_CORRUPT",
            f"Downloaded snapshot failed integrity verification ({len(problems)} problems): {problems[:5]}",
            hint="Delete the affected blobs from the HF cache (see logs) and re-run stage 02; downloads resume.",
            details={"problems": problems},
        )
    return hashes
