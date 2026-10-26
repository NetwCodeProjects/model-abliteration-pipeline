"""Stage 02 - acquire: pin the exact model revision, download through the official HF cache,
verify every file against published hashes, and emit the immutable source manifest.

Also fetches inputs that later stages need (reproduce.json, pinned prompt datasets, imatrix text)
so that their identity is recorded before any GPU time is spent.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from .. import configspec
from ..adapters.heretic import dataset_specs, native_template
from ..adapters.hf_hub import COMMIT_RE, normalize_model_ref, validate_revision, verify_snapshot
from ..artifacts.manifest import build_manifest
from ..artifacts.safetensors_meta import dtype_summary, tensor_map
from ..config import PROJECT_ROOT
from ..errors import ExitCode, PipelineError
from ..fsutil import (
    atomic_write_json,
    disk_free_bytes,
    file_inventory,
    human_bytes,
    make_read_only,
    quick_fingerprint,
    read_json,
    sha256_file,
    sha256_obj,
)
from ..telemetry import utc_now
from .base import Stage, StageContext


def template_path(config) -> Path:
    configured = config.get("abliteration.heretic_config")
    return Path(configured) if configured else PROJECT_ROOT / "configs" / "heretic" / f"{config.get('abliteration.experiment')}.toml"


def hub_version() -> str | None:
    try:
        import importlib.metadata as md  # noqa: PLC0415

        return "huggingface_hub " + md.version("huggingface_hub")
    except Exception:  # noqa: BLE001
        return None


def parse_hf_uri(uri: str) -> tuple[str, str, str, str]:
    """hf://[datasets/]<repo>@<rev>/<path> -> (repo_type, repo, rev, path)."""
    rest = uri[len("hf://") :]
    repo_type = "model"
    if rest.startswith("datasets/"):
        repo_type, rest = "dataset", rest[len("datasets/") :]
    if "@" not in rest:
        raise PipelineError(ExitCode.CONFIG, "E_HF_URI", f"{uri}: expected hf://[datasets/]<org>/<name>@<revision>/<file>")
    repo, tail = rest.split("@", 1)
    if "/" not in tail:
        raise PipelineError(ExitCode.CONFIG, "E_HF_URI", f"{uri}: missing file path after the revision")
    rev, path = tail.split("/", 1)
    if ".." in path.split("/"):
        raise PipelineError(ExitCode.CONFIG, "E_HF_URI", f"{uri}: '..' is not allowed")
    return repo_type, normalize_model_ref(repo), validate_revision(rev), path


class AcquireStage(Stage):
    id = "02"
    name = "acquire"
    title = "Acquire and verify inputs"
    config_keys = (
        "model.url",
        "model.revision",
        "model.local_path",
        "abliteration.enabled",
        "abliteration.experiment",
        "abliteration.heretic_config",
        "abliteration.reproduce",
        "evaluation.qualification",
        "evaluation.heldout_refusal_prompts",
        "evaluation.heldout_harmless_prompts",
        "evaluation.extended_kl",
        "quantization.input",
        "quantization.input_path",
        "quantization.gguf.imatrix_dataset",
    )

    def dependencies(self, config) -> tuple[str, ...]:
        return ("00",)

    def extra_fingerprint(self, run) -> dict[str, Any]:
        path = template_path(run.config)
        return {"template_sha256": sha256_file(path) if run.config.get("abliteration.enabled") and path.is_file() else None}

    def target_outputs(self, config) -> list[str]:
        out = []
        if config.get("model.url"):
            out.append(f"workspace/models/source/{config.get('model.url').replace('/', '--')}/<sha>.json (+ HF cache snapshot, sha256-verified)")
        else:
            out.append(f"{config.get('model.local_path')} (hashed in place, provenance: supplied)")
        out.append("workspace/manifests/<source artifact id>.json")
        return out

    def validate_outputs(self, run, receipt) -> list[str]:
        problems = super().validate_outputs(run, receipt)
        diag = receipt.get("diagnostics", {})
        for role in ("source", "quant_input"):
            path, fingerprint = diag.get(f"{role}_path"), diag.get(f"{role}_quick_fp")
            if not (path and fingerprint):
                continue
            if not Path(path).exists():
                problems.append(f"{role} files are gone: {path}")
            elif sha256_obj(quick_fingerprint(path)) != fingerprint:
                problems.append(f"{role} files changed since verification (immutability violated): {path}")
        return problems

    # ------------------------------------------------------------------ helpers
    def _register_local(self, ctx: StageContext, path: Path, role: str) -> dict[str, Any]:
        if not path.is_dir():
            raise PipelineError(ExitCode.ACQUIRE, "E_LOCAL_MODEL_INVALID", f"{role}: {path} is not a directory")
        missing = [n for n in ("config.json",) if not (path / n).is_file()]
        if missing or not list(path.glob("*.safetensors")):
            raise PipelineError(
                ExitCode.ACQUIRE,
                "E_LOCAL_MODEL_INVALID",
                f"{path} must contain config.json and *.safetensors files (found missing: {missing or 'safetensors'})",
                hint="Point --model-path at a Hugging Face-format checkpoint directory.",
            )
        ctx.info(f"Hashing local model {path} (this reads every file once)")
        files = [e.to_dict() for e in file_inventory(path)]
        summary = dtype_summary(tensor_map(path))
        manifest = build_manifest(
            run_id=ctx.run.run_id,
            model_family=ctx.config.get("model.family"),
            source_model=None,
            source_revision=None,
            parent_artifact_id=None,
            kind="original",
            method="supplied-local-directory",
            config_hash=sha256_obj(ctx.config.subset(list(self.config_keys))),
            tool_revision=None,
            precision=summary["precision"],
            fmt="safetensors",
            uri=str(path),
            files=files,
            validation={"integrity_passed": True, "inference_passed": None, "modalities_tested": []},
            extensions={"provenance": "supplied", "integrity_basis": "hashed locally; no upstream hashes to compare", "dtype_summary": summary},
        )
        ctx.run.registry.register(manifest, role, path, self.id, sidecar=False)
        ctx.output_artifacts.append(manifest["artifact_id"])
        ctx.diagnostics[f"{role}_path"] = str(path)
        ctx.diagnostics[f"{role}_quick_fp"] = sha256_obj(quick_fingerprint(path))
        return manifest

    def _acquire_hub_model(self, ctx: StageContext) -> dict[str, Any]:
        run, config = ctx.run, ctx.config
        repo = config.get("model.url")
        hub = run.hub()
        info = hub.resolve(repo, config.get("model.revision"))
        if not COMMIT_RE.match(info.sha or ""):
            raise PipelineError(ExitCode.ACQUIRE, "E_HF_REVISION", f"Hub returned a non-commit revision {info.sha!r} for {repo}")
        ctx.info(f"Resolved {repo}@{config.get('model.revision') or 'default'} -> {info.sha}")
        record_dir = run.workspace / "models" / "source" / repo.replace("/", "--")
        record_path = record_dir / f"{info.sha}.json"
        record = read_json(record_path) if record_path.exists() else None
        if record and Path(record["snapshot_path"]).exists() and sha256_obj(quick_fingerprint(record["snapshot_path"])) == record["quick_fp"]:
            ctx.info("Verified snapshot already present; skipping download and re-hash")
            snapshot = Path(record["snapshot_path"])
        else:
            total = info.total_size
            free = disk_free_bytes(run.cache_root)
            if total is not None and free < total + float(config.get("environment.min_free_disk_gb")) * 1024**3:
                raise PipelineError(
                    ExitCode.ACQUIRE,
                    "E_DISK_INSUFFICIENT",
                    f"{repo} needs {human_bytes(total)} but only {human_bytes(free)} is free under {run.cache_root}",
                )
            ctx.info(f"Downloading {repo}@{info.sha[:12]} ({human_bytes(total)}) via the HF cache (resumable)")
            snapshot = hub.snapshot(repo, info.sha)
            run.check_interrupt()
            hashes = verify_snapshot(snapshot, info, progress=lambda p: ctx.logger.debug("acquire.hash", p))
            try:
                touched = make_read_only(snapshot)
                ctx.diagnostics["read_only_files"] = touched
            except OSError as exc:
                ctx.warn(f"Could not mark source blobs read-only: {exc}")
            record = {
                "repo": repo,
                "sha": info.sha,
                "requested_revision": config.get("model.revision"),
                "snapshot_path": str(snapshot),
                "files": [{"relative_path": f.path, "size_bytes": f.size, "sha256": hashes[f.path]} for f in info.files],
                "total_size": total,
                "gated": info.gated,
                "license": (info.card_data or {}).get("license"),
                "verified_utc": utc_now(),
                "quick_fp": sha256_obj(quick_fingerprint(snapshot)),
            }
            atomic_write_json(record_path, record)
        link = record_dir / f"{repo.split('/')[-1]}@{info.sha[:12]}"
        if not link.exists():
            try:
                os.symlink(snapshot, link, target_is_directory=True)
            except OSError:
                pass
        summary = dtype_summary(tensor_map(snapshot))
        manifest = build_manifest(
            run_id=run.run_id,
            model_family=config.get("model.family"),
            source_model=repo,
            source_revision=info.sha,
            parent_artifact_id=None,
            kind="original",
            method="huggingface_hub.snapshot_download (sha256/git-blob verified)",
            config_hash=sha256_obj(config.subset(list(self.config_keys))),
            tool_revision=hub_version(),
            precision=summary["precision"],
            fmt="safetensors",
            uri=str(snapshot),
            files=record["files"],
            validation={"integrity_passed": True, "inference_passed": None, "modalities_tested": []},
            extensions={
                "hub_uri": f"hf://{repo}@{info.sha}",
                "license": record.get("license"),
                "dtype_summary": summary,
                "storage": "Hugging Face cache snapshot (symlinks to read-only blobs)",
            },
        )
        run.registry.register(manifest, "source", record_dir / info.sha, self.id, sidecar=False)
        ctx.output_artifacts.append(manifest["artifact_id"])
        ctx.add_output(record_path, "source_record")
        ctx.diagnostics.update(
            source_repo=repo,
            source_revision=info.sha,
            source_path=str(snapshot),
            source_quick_fp=record["quick_fp"],
            source_total_size=record["total_size"],
        )
        return manifest

    def _fetch_reproduce(self, ctx: StageContext) -> None:
        run, rep = ctx.run, ctx.config.get("abliteration.reproduce")
        dest = run.run_dir / "configs" / "reproduce-source.json"
        info: dict[str, Any] = {"exact_file_available": False}
        if rep["source"]:
            src = rep["source"]
            if src.startswith("hf://"):
                repo_type, repo, rev, path = parse_hf_uri(src)
                resolved = run.hub().resolve(repo, rev, repo_type=repo_type)
                shutil.copyfile(run.hub().download_file(repo, path, resolved.sha, repo_type=repo_type), dest)
                info.update(origin=f"hf://{repo}@{resolved.sha}/{path}")
            elif src.startswith("https://"):
                with urllib.request.urlopen(src, timeout=60) as response:  # noqa: S310 - https only, user-configured
                    dest.write_bytes(response.read(10 * 1024 * 1024))
                info.update(origin=src)
            else:
                shutil.copyfile(src, dest)
                info.update(origin=str(Path(src).resolve()))
            info.update(exact_file_available=True)
        elif rep["reference_repo"]:
            resolved = run.hub().resolve(rep["reference_repo"], rep["reference_revision"])
            info.update(reference_repo=rep["reference_repo"], reference_revision=resolved.sha)
            if any(f.path == "reproduce/reproduce.json" for f in resolved.files):
                shutil.copyfile(run.hub().download_file(rep["reference_repo"], "reproduce/reproduce.json", resolved.sha), dest)
                info.update(exact_file_available=True, origin=f"hf://{rep['reference_repo']}@{resolved.sha}/reproduce/reproduce.json")
            else:
                ctx.warn(
                    f"{rep['reference_repo']}@{resolved.sha[:12]} publishes no reproduce/reproduce.json: exact reproduction is impossible; "
                    "only an explicitly allowed approximate reproduction can proceed (decided in stage 04)."
                )
        if info["exact_file_available"]:
            info["sha256"] = sha256_file(dest)
            ctx.add_output(dest, "reproduce_json", sha256=info["sha256"])
            info["path"] = str(dest)
        ctx.diagnostics["reproduce"] = info

    def _prefetch_datasets(self, ctx: StageContext) -> None:
        config = ctx.config
        specs: list[dict[str, Any]] = []
        if config.get("abliteration.enabled"):
            specs.extend(dataset_specs(native_template(template_path(config))))
            if config.get("abliteration.experiment") == "reproduce":
                specs.extend(
                    [
                        {"dataset": configspec.HARMLESS_DATASET, "commit": configspec.HARMLESS_DATASET_COMMIT},
                        {"dataset": configspec.HARMFUL_DATASET, "commit": configspec.HARMFUL_DATASET_COMMIT},
                    ]
                )
            if config.get("evaluation.qualification") != "off":
                specs.extend(config.get("evaluation.heldout_refusal_prompts") or [])
                if config.get("evaluation.extended_kl") or config.get("evaluation.qualification") == "extended":
                    specs.extend(config.get("evaluation.heldout_harmless_prompts") or [])
        fetched: dict[str, Any] = {}
        for spec in specs:
            name = spec["dataset"]
            if Path(name).exists():
                fetched[name] = {"local": True, "sha256": sha256_file(name) if Path(name).is_file() else None}
                continue
            if not spec.get("commit"):
                ctx.warn(f"Dataset {name} has no pinned commit; Heretic/qualification would load whatever the Hub serves today")
                continue
            key = f"{name}@{spec['commit']}"
            if key in fetched:
                continue
            ctx.run.check_interrupt()
            info = ctx.run.hub().resolve(name, spec["commit"], repo_type="dataset")
            snapshot = ctx.run.hub().snapshot(name, info.sha, repo_type="dataset")
            hashes = verify_snapshot(snapshot, info)
            fetched[key] = {"sha": info.sha, "files": len(hashes), "digest": sha256_obj(sorted(hashes.items()))}
        ctx.diagnostics["datasets"] = fetched

    def _fetch_imatrix(self, ctx: StageContext) -> None:
        spec = ctx.config.get("quantization.gguf.imatrix_dataset")
        if not spec or not ctx.config.get("quantization.enabled"):
            return
        if spec.startswith("hf://"):
            repo_type, repo, rev, path = parse_hf_uri(spec)
            resolved = ctx.run.hub().resolve(repo, rev, repo_type=repo_type)
            local = ctx.run.hub().download_file(repo, path, resolved.sha, repo_type=repo_type)
            origin = f"hf://{'datasets/' if repo_type == 'dataset' else ''}{repo}@{resolved.sha}/{path}"
        else:
            local = Path(spec)
            if not local.is_file():
                raise PipelineError(ExitCode.ACQUIRE, "E_IMATRIX_DATA_MISSING", f"imatrix calibration file not found: {spec}")
            origin = str(local.resolve())
        ctx.diagnostics["imatrix_data"] = {"path": str(local), "origin": origin, "sha256": sha256_file(local)}

    # ------------------------------------------------------------------ execute
    def execute(self, ctx: StageContext) -> None:
        config = ctx.config
        if config.get("model.url"):
            self._acquire_hub_model(ctx)
        else:
            self._register_local(ctx, Path(config.get("model.local_path")), "source")
        if config.get("quantization.enabled") and config.get("quantization.input") == "path":
            manifest = self._register_local(ctx, Path(config.get("quantization.input_path")), "quant_input")
            ctx.diagnostics["quant_input_artifact"] = manifest["artifact_id"]
        if config.get("abliteration.enabled") and config.get("abliteration.experiment") == "reproduce":
            self._fetch_reproduce(ctx)
        self._prefetch_datasets(ctx)
        self._fetch_imatrix(ctx)
