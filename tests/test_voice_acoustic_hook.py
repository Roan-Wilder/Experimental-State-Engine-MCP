from __future__ import annotations

from voice.acoustic import (
    AcousticAnalyzerRegistry,
    BasicAcousticAnalyzer,
    analyze_acoustic_observation,
    build_mock_wav,
    extract_acoustic_features,
)


def test_basic_analyzer_computes_duration_rms_peak():
    wav = build_mock_wav(duration_seconds=0.5, seed=1)
    features = BasicAcousticAnalyzer().analyze(wav)
    assert features["duration_seconds"] > 0
    assert features["rms_energy"] is not None
    assert features["peak_amplitude"] is not None
    assert features["f0_hz"] is None


def test_registry_returns_acoustic_observation_wrapper():
    wav = build_mock_wav(seed=2)
    payload = AcousticAnalyzerRegistry().run(wav)
    assert "features" in payload
    assert "acoustic_observation" in payload
    assert payload["acoustic_observation"]["analyzers"][0]["analyzer"] == "basic"


def test_extract_acoustic_features_backward_compatible():
    wav = build_mock_wav(seed=3)
    features = extract_acoustic_features(wav)
    assert "duration_seconds" in features
    assert "emotion" not in features
    assert "consciousness" not in features


def test_analyze_acoustic_observation_has_no_inference_keys():
    payload = analyze_acoustic_observation(build_mock_wav(seed=4))
    feature_keys = set(payload["features"].keys())
    assert "emotion" not in feature_keys
    assert "consciousness" not in feature_keys
    assert "sentience" not in feature_keys
