"""Shared planning helpers: required components, disk estimates and the --dry-run report.

Nothing in this module writes files, opens network connections, runs subprocesses or imports
GPU libraries; ``describe_plan`` only reads configuration and existing workspace metadata.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import configspec
from .config import COMPATIBILITY_NOTICE, ResolvedConfig
from .environment.base import BASE_REQUIREMENTS, GPU_COMPONENTS
from .environment.detect import HostProbe, detect_profile, lambda_stack_markers
from .errors import PipelineError
from .fsutil import human_bytes

if TYPE_CHECKING:  # pragma: no cover
    from .orchestrator import RunOptions

# Approximate bits per weight (llama.cpp quantize table / docs); used only for estimates.
GGUF_BPW = {
    "q8_0": 8.5,
    "q6_k": 6.56,
    "q5_k_m": 5.7,
    "q5_k_s": 5.55,
    "q5_0": 5.5,
    "q5_1": 6.0,
    "q4_k_m": 4.85,
    "q4_k_s": 4.6,
    "q4_0": 4.5,
    "q4_1": 5.0,
    "iq4_xs": 4.25,
    "iq4_nl": 4.5,
    "q3_k_l": 4.3,
    "q3_k_m": 3.9,
    "q3_k_s": 3.5,
    "iq3_m": 3.66,
    "iq3_s": 3.44,
    "iq3_xs": 3.3,
    "iq3_xxs": 3.06,
    "q2_k": 2.96,
    "q2_k_s": 2.8,
    "iq2_m": 2.7,
    "iq2_s": 2.5,
    "iq2_xs": 2.31,
    "iq2_xxs": 2.06,
    "iq1_m": 1.75,
    "iq1_s": 1.56,
    "tq1_0": 1.69,
    "tq2_0": 2.06,
}


def components_for_config(config: ResolvedConfig) -> set[str]:
    comps: set[str] = set()
    if config.get("abliteration.enabled"):
        comps.add("heretic")
    if config.get("quantization.enabled"):
        formats = config.get("quantization.formats") or []
        if "fp8" in formats:
            comps.add("fp8")
            if config.get("quantization.fp8.verify_backend") == "vllm":
                comps.add("vllm")
        if any(f in configspec.GGUF_TYPES for f in formats):
            comps.add("llama_cpp")
        if formats and config.get("evaluation.comparison.enabled"):
            comps.add("heretic")  # transformers runtime for the comparison baseline (also in quantize-only runs)
    return comps


def needs_gpu(config: ResolvedConfig) -> bool:
    return bool(components_for_config(config) & GPU_COMPONENTS)


def resolved_fp8_scheme(config: ResolvedConfig) -> str:
    scheme = config.get("quantization.fp8.scheme")
    return "fp8_block" if scheme == "auto" else scheme


def known_source_size(config: ResolvedConfig) -> int | None:
    """Size of a previously verified source snapshot recorded in the workspace (read-only lookup)."""
    repo = config.get("model.url")
    if not repo:
        return None
    record_dir = Path(config.get("storage.root")) / "models" / "source" / repo.replace("/", "--")
    revision = config.get("model.revision")
    if not record_dir.is_dir():
        return None
    for record in sorted(record_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if revision in (None, data.get("sha"), data.get("requested_revision")):
            return data.get("total_size")
    return None


def estimate_disk(config: ResolvedConfig, source_bytes: int | None, source_cached: bool = False) -> dict[str, Any]:
    if source_bytes is None:
        return {"total_bytes": None, "items": [], "note": "unknown until stage 02 resolves the model (no network in dry-run)"}
    s = float(source_bytes)
    items: list[tuple[str, float]] = []
    if not source_cached and config.get("model.url"):
        items.append(("source download (HF cache)", s))
    if config.get("abliteration.enabled"):
        items.append(("Heretic candidate checkpoint (merged, ~source size)", s))
        if config.get("storage.link_strategy") == "copy":
            items.append(("selected master copy", s))
    if config.get("quantization.enabled"):
        formats = config.get("quantization.formats") or []
        if "fp8" in formats:
            items.append(("fp8 export (~55% of BF16; estimate)", 0.55 * s))
        gguf = [f for f in formats if f in configspec.GGUF_TYPES]
        if gguf:
            inter = {"bf16": 1.0, "f16": 1.0, "f32": 2.0}[config.get("quantization.gguf.intermediate_type")]
            items.append((f"GGUF {config.get('quantization.gguf.intermediate_type')} intermediate", inter * s))
            for fmt in gguf:
                items.append((f"GGUF {fmt} (~{GGUF_BPW.get(fmt, 8)} bpw)", s * GGUF_BPW.get(fmt, 8.0) / 16.0))
    total = sum(v for _, v in items)
    return {"total_bytes": int(total), "items": [(k, int(v)) for k, v in items], "note": "estimates; actual sizes depend on the model"}


def describe_plan(config: ResolvedConfig, options: RunOptions, probe: HostProbe) -> str:
    from .orchestrator import find_resumable_run, selected_stage_ids  # noqa: PLC0415
    from .stages import STAGES  # noqa: PLC0415

    lines: list[str] = []
    add = lines.append
    add("DRY RUN (no network, downloads, installs, GPU work, model loading or file writes)")
    add(f"Config sources : {', '.join(config.sources)}")
    add(f"Config hash    : {config.config_hash}")
    model = config.get("model.url") or config.get("model.local_path")
    revision = config.get("model.revision") or "default branch"
    add(
        f"Model          : {model}"
        + (f" (revision '{revision}' -> resolved to a commit SHA in stage 02)" if config.get("model.url") else " (local, provenance: supplied)")
    )
    mode = "setup-only" if options.setup_only else ("quantize-only" if not config.get("abliteration.enabled") else "full pipeline")
    add(f"Mode           : {mode}")
    try:
        profile, reason = detect_profile(probe, config.get("environment.profile"))
        add(f"Environment    : {profile} ({reason})")
    except PipelineError as exc:
        add(f"Environment    : UNRESOLVED - {exc.message} (stage 00 would fail; hint: {exc.hint})")
    markers = lambda_stack_markers(probe)
    if markers:
        add(f"Lambda markers : {', '.join(markers)}")
    workspace = Path(config.get("storage.root"))
    run_note = "new run"
    if options.resume_run_id:
        run_note = f"resume {options.resume_run_id} (stored config is authoritative)"
    elif config.get("execution.resume") and not options.new_run:
        found = find_resumable_run(workspace, config.config_hash)
        if found:
            run_note = f"would auto-resume incomplete run {found} (identical config hash)"
    add(f"Run            : {run_note}; workspace {workspace}")
    add("")
    add("Stages:")
    try:
        selected = selected_stage_ids(options)
    except PipelineError as exc:
        add(f"  INVALID STAGE SELECTION: {exc}")
        selected = []
    for stage in STAGES:
        if stage.id not in selected:
            add(f"  {stage.id} {stage.name:20s} not selected")
            continue
        ok, why = stage.applicable(config)
        deps = ",".join(stage.dependencies(config)) or "-"
        state = "run (or reuse a valid receipt)" if ok else f"skip: {why}"
        add(f"  {stage.id} {stage.name:20s} {state}  [deps: {deps}]")
        for target in stage.target_outputs(config) if ok else []:
            add(f"       -> {target}")
    add("")
    comps = components_for_config(config)
    pins = {
        "heretic": f"Heretic {config.get('abliteration.engine_revision') or configspec.HERETIC_PINNED_REVISION} ({config.get('abliteration.engine_repo')})",
        "fp8": f"llm-compressor {config.get('quantization.fp8.llmcompressor_version')} / compressed-tensors {config.get('quantization.fp8.compressed_tensors_version')}",
        "vllm": f"vLLM {config.get('quantization.fp8.vllm_version')} (FP8 loader smoke test)",
        "llama_cpp": f"llama.cpp {config.get('quantization.gguf.llama_cpp_revision') or configspec.LLAMA_CPP_PINNED_REVISION} (source build) + converter venv",
    }
    add("Components     : " + (", ".join(pins[c] for c in sorted(comps)) if comps else "none"))
    host = []
    for req in BASE_REQUIREMENTS:
        if req.needed_for & comps:
            present = shutil.which(req.command) is not None
            host.append(f"{req.name}{'' if present else ' (MISSING' + (', optional' if req.optional else '') + ')'}")
    add("Host tools     : " + (", ".join(host) if host else "none needed"))
    add(f"GPU required   : {'yes' if needs_gpu(config) else 'no'}")
    if config.get("abliteration.enabled"):
        exp = config.get("abliteration.experiment")
        add(
            f"Abliteration   : Heretic experiment={exp} max_trials={config.get('abliteration.max_trials')} "
            f"budget={config.get('abliteration.max_runtime_minutes')} min selection={_selection(config)} "
            f"template={config.get('abliteration.heretic_config') or 'configs/heretic/' + exp + '.toml'}"
        )
        if exp == "reproduce":
            rep = config.get("abliteration.reproduce")
            add(
                f"Reproduction   : source={rep['source'] or '-'} reference={rep['reference_repo'] or '-'} "
                f"allow_approximate={rep['allow_approximate']} (exactness decided in stage 04 from what actually exists)"
            )
        add(f"Qualification  : {config.get('evaluation.qualification')} (independent held-out checks; optimizer metrics are imported and labelled)")
    if config.get("quantization.enabled"):
        formats = config.get("quantization.formats")
        add(f"Quantization   : input={config.get('quantization.input')} formats={', '.join(formats) or 'none'}")
        if "fp8" in formats:
            add(
                f"  fp8          : {resolved_fp8_scheme(config)} via llm-compressor model_free_ptq (data-free), verify={config.get('quantization.fp8.verify_backend')}"
            )
        gguf = [f for f in formats if f in configspec.GGUF_TYPES]
        if gguf:
            imatrix = config.get("quantization.gguf.imatrix_dataset")
            needs = [f for f in gguf if f in configspec.GGUF_TYPES_REQUIRING_IMATRIX]
            add(
                f"  gguf         : {', '.join(gguf)}; imatrix={imatrix or 'none'}"
                + (f"; WILL FAIL without imatrix: {', '.join(needs)}" if needs and not imatrix else "")
            )
            add(f"  multimodal   : preserve={config.get('quantization.gguf.preserve_multimodal')} (mmproj only claimed after a real test)")
        if formats and config.get("evaluation.comparison.enabled"):
            cmp = config.get("evaluation.comparison")
            baseline = {"selected_master": "master", "source": "source model", "path": "input path"}[config.get("quantization.input")]
            add(
                f"  comparison   : {cmp['benign_prompts']} benign + {cmp['heldout_refusal_prompts']} held-out prompts on the {baseline} "
                f"(transformers) and each export; export gates: {config.get('selection.export_gates.mode')}"
            )
    source_size = known_source_size(config)
    estimate = estimate_disk(config, source_size)
    if estimate["total_bytes"] is None:
        add(f"Disk estimate  : unknown ({estimate['note']})")
    else:
        add(f"Disk estimate  : ~{human_bytes(estimate['total_bytes'])} + buffer {config.get('environment.disk_buffer_gb')} GB ({estimate['note']})")
        for label, size in estimate["items"]:
            add(f"    {label}: ~{human_bytes(size)}")
    add("")
    add("Compatibility notice:")
    add("  " + COMPATIBILITY_NOTICE)
    add("")
    add("Nothing was downloaded, installed, executed on a GPU, or written.")
    return "\n".join(lines)


def _selection(config: ResolvedConfig) -> str:
    if config.get("abliteration.trial_id") is not None:
        return f"manual trial {config.get('abliteration.trial_id')}"
    if config.get("abliteration.pareto_trial_index") is not None:
        return f"manual Pareto position {config.get('abliteration.pareto_trial_index')}"
    return (
        f"{config.get('selection.policy')} (refusal <= {config.get('selection.max_optimizer_refusal_rate')}, KL <= {config.get('selection.max_optimizer_kl')})"
    )
