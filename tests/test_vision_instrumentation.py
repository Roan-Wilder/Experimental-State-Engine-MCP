from __future__ import annotations

from pathlib import Path

import pytest

from instrumentation.facade import InstrumentedStateFacade
from instrumentation.session import TrialSession
from instrumentation.store import ObservationStore
from instrumentation.vision.capability import detect_vision_capability
from instrumentation.vision.data_url import build_data_url, parse_data_url
from instrumentation.vision.hashing import sha256_hex
from instrumentation.vision.metadata import extract_image_metadata, minimal_png_bytes
from instrumentation.vision.mime import validate_mime_type
from instrumentation.vision.registry import ImageProvenanceStore
from instrumentation.vision.register import register_image_bytes, register_stimulus_image
from instrumentation.vision.request_builder import LMStudioVisionRequestBuilder
from instrumentation.vision.stimuli import resolve_stimulus_bytes
from instrumentation.vision.types import VisionRequestContext
from research.visual_trial_runner import VisualTrialRunner
from state.experimental_provider import ExperimentalStateProvider
from state.snapshots import SnapshotStore


@pytest.fixture
def png_bytes() -> bytes:
    return minimal_png_bytes(width=4, height=4)


@pytest.fixture
def tmp_registry(tmp_path: Path) -> ImageProvenanceStore:
    return ImageProvenanceStore(tmp_path / "image_provenance.jsonl")


def test_build_vision_chat_request_with_image_item(tmp_path: Path):
    registry = ImageProvenanceStore(tmp_path / "prov.jsonl")
    reg = register_stimulus_image(
        stimulus_key="image_1_box_b",
        experiment_id="exp-test",
        stage="visual_prediction",
        registry=registry,
    )
    builder = LMStudioVisionRequestBuilder(registry)
    context = VisionRequestContext.new(experiment_id="exp-test", stage="visual_prediction")
    request = builder.build_chat_request(
        text="Where is the token?",
        image_ids=[reg["image_id"]],
        context=context,
    )
    assert "messages" in request
    content = request["messages"][0]["content"]
    assert any(item.get("type") == "image" for item in content)


def test_data_url_round_trip(png_bytes: bytes):
    url = build_data_url(png_bytes, "image/png")
    parsed = parse_data_url(url)
    assert parsed["mime_type"] == "image/png"
    assert parsed["data"] == png_bytes


def test_mime_validation_allowed_and_rejected(png_bytes: bytes):
    assert validate_mime_type("image/png", png_bytes) is None
    bad = validate_mime_type("image/png", b"not-an-image")
    assert bad is not None


def test_sha256_hash_stable(png_bytes: bytes):
    assert sha256_hex(png_bytes) == sha256_hex(png_bytes)


def test_extract_png_dimensions_synthetic(png_bytes: bytes):
    meta = extract_image_metadata(png_bytes, "image/png")
    assert meta["width"] == 4
    assert meta["height"] == 4


def test_duplicate_image_detected_by_hash(tmp_registry: ImageProvenanceStore, png_bytes: bytes):
    first = register_image_bytes(
        data=png_bytes,
        mime_type="image/png",
        source="tests/a.png",
        experiment_id="exp",
        stage="visual_prediction",
        registry=tmp_registry,
    )
    second = register_image_bytes(
        data=png_bytes,
        mime_type="image/png",
        source="tests/b.png",
        experiment_id="exp",
        stage="visual_contradiction",
        registry=tmp_registry,
    )
    assert second["duplicate_of_image_id"] == first["image_id"]


def test_image_linked_to_trial_and_stage(tmp_path: Path):
    resolved = resolve_stimulus_bytes("image_1_box_b")
    assert resolved.get("data")
    reg = register_stimulus_image(
        stimulus_key="image_1_box_b",
        experiment_id="exp-visual",
        stage="visual_prediction",
        trial_id="trial-123",
        registry=ImageProvenanceStore(tmp_path / "prov.jsonl"),
    )
    assert reg["trial_id"] == "trial-123"
    assert reg["stage"] == "visual_prediction"
    assert reg["sha256"]
    assert reg["width"] >= 1


def test_multiple_images_in_one_request(tmp_path: Path):
    registry = ImageProvenanceStore(tmp_path / "prov.jsonl")
    ids = []
    for key in ("image_1_box_b", "image_2_box_c"):
        reg = register_stimulus_image(
            stimulus_key=key,
            experiment_id="exp",
            stage="visual_prediction",
            registry=registry,
        )
        ids.append(reg["image_id"])
    builder = LMStudioVisionRequestBuilder(registry)
    context = VisionRequestContext.new(experiment_id="exp", stage="visual_prediction")
    request = builder.build_chat_request(text="Compare", image_ids=ids, context=context)
    assert len(request["image_ids"]) == 2


def test_missing_image_returns_structured_error(tmp_path: Path):
    registry = ImageProvenanceStore(tmp_path / "prov.jsonl")
    builder = LMStudioVisionRequestBuilder(registry)
    context = VisionRequestContext.new(experiment_id="exp", stage="visual_prediction")
    result = builder.build_chat_request(
        text="Missing",
        image_ids=["00000000-0000-4000-8000-000000000001"],
        context=context,
    )
    assert result["error"] == "image_not_found"


def test_unsupported_format_rejected():
    result = validate_mime_type("image/png", b"GIF89a")
    assert result is not None


def test_vision_capability_detection(monkeypatch):
    monkeypatch.setenv("AI_ECO_LAB_VISION_ENABLED", "1")
    assert detect_vision_capability() == "supported"
    monkeypatch.delenv("AI_ECO_LAB_VISION_ENABLED", raising=False)
    assert detect_vision_capability({"vision_capable": True}) == "supported"


def test_provenance_preserved_in_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    provider = ExperimentalStateProvider(snapshots=SnapshotStore(tmp_path / "snapshots"))
    facade = InstrumentedStateFacade(
        provider=provider,
        session=TrialSession(experiment_id="exp", trial_id="trial-1", source="test"),
        store=ObservationStore(tmp_path / "observations.jsonl"),
        image_store=ImageProvenanceStore(tmp_path / "prov.jsonl"),
        protocol={"vision_capable": True},
    )
    reg = register_stimulus_image(
        stimulus_key="image_1_box_b",
        experiment_id="exp",
        stage="visual_prediction",
        trial_id="trial-1",
        registry=facade.image_store,
    )
    ctx = VisionRequestContext.new(experiment_id="exp", stage="visual_prediction", trial_id="trial-1")
    ctx.image_ids = [reg["image_id"]]
    ctx.image_provenance = [reg]
    obs = facade.capture_observation(stage="visual_prediction", trial_id="trial-1", vision_context=ctx)
    assert obs["model_io"]["vision"]["images"] == [reg["image_id"]]


def test_visual_description_not_in_engine_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    provider = ExperimentalStateProvider(snapshots=SnapshotStore(tmp_path / "snapshots"))
    facade = InstrumentedStateFacade(
        provider=provider,
        session=TrialSession(experiment_id="exp", trial_id="trial-1", source="test"),
        store=ObservationStore(tmp_path / "observations.jsonl"),
    )
    engine_before = dict(provider.current_state)
    payload = facade.record_visual_response(
        visual_description="I can see that the object moved.",
        prediction="token is in Box C",
        confidence=0.7,
        request_id="req-1",
        trial_id="trial-1",
    )
    assert payload["behavioral"]["visual_description"].startswith("I can see")
    assert dict(provider.current_state) == engine_before
    assert "visual_description" not in payload.get("engine_state", {})


def test_visual_trial_runner_simulation_batch(tmp_path: Path):
    protocol = Path(__file__).resolve().parents[1] / "research" / "protocols" / "visual_prediction_v1.yaml"
    runner = VisualTrialRunner(
        experiment_id="exp-visual-test",
        protocol_path=protocol,
        raw_log_path=tmp_path / "trials.jsonl",
        observations_path=tmp_path / "observations.jsonl",
        image_provenance_path=tmp_path / "image_provenance.jsonl",
    )
    completed = runner.run_batch()
    assert len(completed) == 10
    assert all(t.get("outcome") == "completed" for t in completed)
    assert all(t.get("image_ids") for t in completed)
