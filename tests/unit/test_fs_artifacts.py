"""Atomic writes, hashing/inventories, locks, path safety, safetensors/GGUF readers, manifests."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest
from conftest import ROOT, make_tiny_model

from llm_surgeon.artifacts.gguf_meta import read_metadata, write_minimal_gguf
from llm_surgeon.artifacts.manifest import ArtifactRegistry, artifact_id_for, build_manifest, check_manifest
from llm_surgeon.artifacts.safetensors_meta import dtype_summary, graft_tensors, read_header, tensor_diff, tensor_map, write_safetensors
from llm_surgeon.config import load_yaml
from llm_surgeon.errors import PipelineError
from llm_surgeon.fsutil import (
    FileLock,
    atomic_write_json,
    ensure_within,
    file_inventory,
    inventory_digest,
    link_or_copy_tree,
    quick_fingerprint,
    sha256_file,
)


def test_yaml_12_booleans(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("execution:\n  yes: false\nevaluation:\n  qualification: off\nflag: true\n", encoding="utf-8")
    data = load_yaml(path)
    assert data == {"execution": {"yes": False}, "evaluation": {"qualification": "off"}, "flag": True}


def test_atomic_write_replaces_without_temp_leftovers(tmp_path):
    target = tmp_path / "a" / "b.json"
    atomic_write_json(target, {"x": 1})
    atomic_write_json(target, {"x": 2})
    assert json.loads(target.read_text()) == {"x": 2}
    assert [p.name for p in target.parent.iterdir()] == ["b.json"]


def test_inventory_hashes_every_shard_and_is_order_independent(tmp_path):
    model = make_tiny_model(tmp_path / "m")
    inv = file_inventory(model)
    names = [e.relative_path for e in inv]
    assert names == sorted(names) and "model-00002-of-00002.safetensors" in names
    assert all(e.sha256 == sha256_file(model / e.relative_path) for e in inv)
    assert inventory_digest(inv) == inventory_digest(list(reversed(inv)))
    assert artifact_id_for(inv).startswith("art-") and len(artifact_id_for(inv)) == 28


def test_ensure_within_rejects_traversal(tmp_path):
    (tmp_path / "ws").mkdir()
    assert ensure_within(tmp_path / "ws", tmp_path / "ws" / "a" / "b")
    with pytest.raises(PipelineError) as err:
        ensure_within(tmp_path / "ws", tmp_path / "ws" / ".." / "etc")
    assert err.value.error_id == "E_PATH_ESCAPE"


def test_file_lock_blocks_live_owner_and_reclaims_stale(tmp_path):
    lock_path = tmp_path / ".lock"
    with FileLock(lock_path, "run"):
        with pytest.raises(PipelineError) as err:
            FileLock(lock_path, "run").acquire()
        assert err.value.error_id == "E_LOCK_HELD"
    assert not lock_path.exists()
    lock_path.write_text(json.dumps({"host": socket.gethostname(), "pid": 2**22 + 12345, "since": 0}))
    with FileLock(lock_path, "run"):
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
    lock_path.write_text(json.dumps({"host": "another-host", "pid": 1, "since": 0}))
    with pytest.raises(PipelineError):
        FileLock(lock_path, "run").acquire()


def test_file_lock_of_a_killed_process_is_stale_even_while_a_handle_is_open(tmp_path):
    # Windows keeps a terminated process openable while any handle exists (here: the Popen object),
    # found when a killed pipeline's lock blocked the resumed run.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert FileLock._pid_alive(proc.pid)
    finally:
        proc.kill()
        proc.wait()
    lock_path = tmp_path / ".lock"
    lock_path.write_text(json.dumps({"host": socket.gethostname(), "pid": proc.pid, "since": 0}))
    with FileLock(lock_path, "study", error_id="E_STUDY_LOCKED"):
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
    lock_file = FileLock(lock_path, "study", error_id="E_STUDY_LOCKED").acquire()
    with pytest.raises(PipelineError) as err:
        FileLock(lock_path, "study", error_id="E_STUDY_LOCKED").acquire()
    lock_file.release()
    assert err.value.error_id == "E_STUDY_LOCKED"


def test_link_or_copy_tree_and_quick_fingerprint(tmp_path):
    src = make_tiny_model(tmp_path / "src")
    semantics = link_or_copy_tree(src, tmp_path / "dst")
    assert semantics in ("hardlink", "copy")
    assert quick_fingerprint(src).keys() == quick_fingerprint(tmp_path / "dst").keys()
    with pytest.raises(PipelineError):
        link_or_copy_tree(src, tmp_path / "dst")


def test_safetensors_header_dtype_and_diff(tmp_path):
    model = make_tiny_model(tmp_path / "m")
    header = read_header(model / "model-00002-of-00002.safetensors")
    assert header["mtp.fc.weight"]["dtype"] == "BF16"
    tensors = tensor_map(model)
    assert dtype_summary(tensors)["precision"] == "bf16"
    other = {k: v for k, v in tensors.items() if not k.startswith("mtp.")}
    diff = tensor_diff(tensors, other)
    assert diff["missing_count"] == 2 and diff["missing_prefixes"] == {"mtp": 2}


def test_mixed_precision_is_reported_honestly(tmp_path):
    write_safetensors(tmp_path / "model.safetensors", {"a": ("BF16", [2], b"\0" * 4), "norm": ("F32", [1], b"\0" * 4)})
    summary = dtype_summary(tensor_map(tmp_path))
    assert summary["precision"].startswith("mixed:") and "bf16" in summary["precision"]


def test_graft_copies_bytes_verbatim_and_leaves_hardlinked_source_untouched(tmp_path):
    source = make_tiny_model(tmp_path / "source")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    tensors = tensor_map(source)
    keep = {}
    for name, meta in tensors.items():
        if name.startswith("mtp."):
            continue
        header = read_header(source / meta["file"])
        with open(source / meta["file"], "rb") as handle:
            handle.seek(8 + header["__header_size__"] + meta["data_offsets"][0])
            keep[name] = (meta["dtype"], meta["shape"], handle.read(meta["data_offsets"][1] - meta["data_offsets"][0]))
    write_safetensors(candidate / "model.safetensors", keep)
    (candidate / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {k: "model.safetensors" for k in keep}}))
    master = tmp_path / "master"
    link_or_copy_tree(candidate, master)
    result = graft_tensors(source, master, ["mtp."])
    assert sorted(result["grafted"]) == ["mtp.fc.weight", "mtp.layers.0.mlp.down_proj.weight"]
    assert tensor_diff(tensor_map(source), tensor_map(master))["missing_count"] == 0
    assert "mtp.fc.weight" not in json.loads((candidate / "model.safetensors.index.json").read_text())["weight_map"]
    # byte-identical tensor payload
    src_meta, dst_meta = tensor_map(source)["mtp.fc.weight"], tensor_map(master)["mtp.fc.weight"]

    def payload(directory, meta):
        with open(directory / meta["file"], "rb") as handle:
            handle.seek(8 + meta["header_size"] + meta["data_offsets"][0])
            return handle.read(meta["data_offsets"][1] - meta["data_offsets"][0])

    assert payload(source, src_meta) == payload(master, dst_meta)


def test_gguf_metadata_round_trip(tmp_path):
    path = tmp_path / "m.gguf"
    write_minimal_gguf(path, {"general.architecture": "qwen35", "tokenizer.ggml.tokens": ["a", "b", "c"], "general.alignment": 32})
    meta = read_metadata(path)
    assert meta["kv"]["general.architecture"] == "qwen35"
    assert meta["kv"]["tokenizer.ggml.tokens"] == {"array_length": 3, "preview": ["a", "b", "c"]}
    (tmp_path / "bad.gguf").write_bytes(b"NOPE")
    with pytest.raises(PipelineError):
        read_metadata(tmp_path / "bad.gguf")


def manifest_kwargs(files):
    return dict(
        run_id="r1",
        model_family="tiny",
        source_model="org/tiny",
        source_revision="a" * 40,
        parent_artifact_id=None,
        kind="original",
        method="test",
        config_hash="b" * 64,
        tool_revision=None,
        precision="bf16",
        fmt="safetensors",
        uri="/x",
        files=files,
    )


def test_manifest_round_trip_and_schema(tmp_path):
    model = make_tiny_model(tmp_path / "m")
    files = [e.to_dict() for e in file_inventory(model)]
    manifest = build_manifest(**manifest_kwargs(files))
    assert manifest["tokenizer"]["hash"] and manifest["processor"]["hash"]
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "artifact.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(manifest, schema)
    reloaded = json.loads(json.dumps(manifest))
    check_manifest(reloaded)
    assert build_manifest(**manifest_kwargs(list(reversed(files))))["artifact_id"] == manifest["artifact_id"]


def test_manifest_refuses_placeholders(tmp_path):
    with pytest.raises(PipelineError):
        build_manifest(**manifest_kwargs([{"relative_path": "a", "size_bytes": 1, "sha256": None}]))
    bad = manifest_kwargs([{"relative_path": "a", "size_bytes": 1, "sha256": "c" * 64}])
    bad["source_revision"] = "main"
    with pytest.raises(PipelineError):
        build_manifest(**bad)
    bad = manifest_kwargs([{"relative_path": "a", "size_bytes": 1, "sha256": "c" * 64}])
    bad["kind"] = "finetune"
    with pytest.raises(PipelineError):
        build_manifest(**bad)


def test_registry_register_attach_and_id_guard(tmp_path):
    from llm_surgeon.artifacts import assessments as A

    reg = ArtifactRegistry(tmp_path / "run", tmp_path / "ws")
    files = [{"relative_path": "a", "size_bytes": 1, "sha256": "c" * 64}]
    manifest = build_manifest(**manifest_kwargs(files))
    target = tmp_path / "ws" / "exports" / "x"
    reg.register(manifest, "export:test", target, "08")
    assert (tmp_path / "ws" / "manifests" / f"{manifest['artifact_id']}.json").exists()
    assert target.with_name("x.manifest.json").exists()
    loader = {"inference_passed": True, "modalities_tested": ["text"], "backend": "b", "backend_revision": "b 1", "tested_context": 64}
    record = A.new_record(
        artifact_id=manifest["artifact_id"],
        role="export:test",
        run_id="r1",
        stage_id="09",
        kind="verification",
        results={},
        checks=[A.check("loader", "passed", True)],
        derives={"validation": {"integrity_passed": True}, "inference": loader},
    )
    reg.attach("export:test", record)
    stored = reg.get("export:test")["manifest_data"]
    assert stored["validation"]["inference_passed"] is True and stored["schema_version"] == "1.1"
    assert stored["derived_status"] == {"validation": record["assessment_id"], "inference": record["assessment_id"], "evaluation": None}
    other = build_manifest(**manifest_kwargs([{"relative_path": "a", "size_bytes": 1, "sha256": "d" * 64}]))
    with pytest.raises(PipelineError) as err:
        reg.attach("export:test", {**record, "artifact_id": other["artifact_id"]})
    assert err.value.error_id == "E_ARTIFACT_UNKNOWN"
