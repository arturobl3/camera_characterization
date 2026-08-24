"""Pure-function tests for the basler backend helpers (no hardware)."""

import numpy as np
import pytest

from camchar.backends.basler import (
    _BAYER_FORMATS,
    _KNOWN_MODELS,
    _model_key,
    _select_pixel_format,
)
from camchar.analyze import (
    _cfa_labels,
    _cfa_phases,
    _dataset_cfa_format,
    dsnu1288_core,
)


def test_model_key_ace_mono():
    assert _model_key("acA1920-155um") == "acA1920-155"
    assert _model_key("acA2500-14gm") == "acA2500-14"


def test_model_key_ace2_color():
    # ace 2 names carry fps/interface/color/line in the suffix
    assert _model_key("a2A3536-31ucBAS") == "a2A3536"
    assert "a2A3536" in _KNOWN_MODELS


def test_select_pixel_format_prefers_raw_bayer():
    symbolics = (
        "Mono8",
        "Mono10p",
        "Mono12",
        "RGB8",
        "BayerRG8",
        "BayerRG12p",
        "BayerRG12",
    )
    assert _select_pixel_format(symbolics) == "BayerRG12"


def test_select_pixel_format_packed_bayer_excluded():
    # packed variants must not be selected: pypylon would not deliver a plain
    # uint16 array with DN12 in the low bits
    fmt = _select_pixel_format(("Mono12", "BayerRG12p"))
    assert fmt == "Mono12"
    assert all(f.endswith("p") is False for f in _BAYER_FORMATS)


def test_select_pixel_format_error_without_candidates():
    with pytest.raises(RuntimeError):
        _select_pixel_format(("Mono8", "RGB8", "YCbCr422_8"))


def test_cfa_phases_split():
    img = np.arange(16, dtype=np.float64).reshape(4, 4)
    parts = _cfa_phases(img)
    assert [p.shape for p in parts] == [(2, 2)] * 4
    # BayerRG layout: R at (0,0), G1 at (0,1), G2 at (1,0), B at (1,1)
    np.testing.assert_array_equal(parts[0], [[0.0, 2.0], [8.0, 10.0]])
    np.testing.assert_array_equal(parts[1], [[1.0, 3.0], [9.0, 11.0]])
    np.testing.assert_array_equal(parts[2], [[4.0, 6.0], [12.0, 14.0]])
    np.testing.assert_array_equal(parts[3], [[5.0, 7.0], [13.0, 15.0]])


def test_dsnu_cfa_removes_pattern_inflation():
    """A CFA-periodic dark pattern inflates DSNU without cfa=True."""
    rng = np.random.default_rng(7)
    base = rng.normal(200.0, 5.0, size=(64, 64))
    pattern = np.zeros((64, 64))
    pattern[0::2, 0::2] = 30.0  # e.g. red pixels sit higher in the dark
    img = base + pattern
    # temporal residual removed by both paths: tvar/n = 25/4
    plain = dsnu1288_core(img, tvar=25.0, n=4, cfa=False)
    cfa = dsnu1288_core(img, tvar=25.0, n=4, cfa=True)
    # without decimation the CFA period survives the Section 8.1 highpass
    assert plain > 10.0
    # with per-phase pooling only the true per-pixel noise remains (~4-5)
    assert cfa < 6.0
    assert cfa < 0.5 * plain


def test_dsnu_cfa_matches_single_phase_on_uniform_image():
    rng = np.random.default_rng(3)
    img = rng.normal(100.0, 4.0, size=(32, 32))
    plain = dsnu1288_core(img, tvar=16.0, n=4, cfa=False)
    cfa = dsnu1288_core(img, tvar=16.0, n=4, cfa=True)
    assert abs(plain - cfa) < 0.05 * max(plain, 1e-9)


def test_cfa_labels_follow_pixel_format_layout():
    assert _cfa_labels("BayerRG12") == ("R", "G1", "G2", "B")
    assert _cfa_labels("BayerGR12") == ("G1", "R", "B", "G2")
    assert _cfa_labels("BayerGB12") == ("G1", "B", "R", "G2")
    assert _cfa_labels("BayerBG12") == ("B", "G1", "G2", "R")
    # unknown / non-Bayer formats fall back to the RGGB default
    assert _cfa_labels("") == ("R", "G1", "G2", "B")
    assert _cfa_labels("Mono12") == ("R", "G1", "G2", "B")


def test_dataset_cfa_format_scans_all_metadata_entries(tmp_path):
    dark = tmp_path / "dark"
    flat = tmp_path / "flat"
    dark.mkdir()
    flat.mkdir()
    # early entry without pixel_format, later entry with it
    (dark / "metadata.json").write_text(
        '[{"model": "a2A3536-31ucBAS"}, '
        '{"model": "a2A3536-31ucBAS", "pixel_format": "BayerRG12"}]'
    )
    assert _dataset_cfa_format(tmp_path) == "BayerRG12"
    # the latest Bayer entry wins (labels follow it)
    (dark / "metadata.json").write_text(
        '[{"pixel_format": "BayerRG12"}, {"pixel_format": "Mono12"}]'
    )
    assert _dataset_cfa_format(tmp_path) == "BayerRG12"
    (dark / "metadata.json").write_text(
        '[{"pixel_format": "BayerRG12"}, {"pixel_format": "BayerGB12"}]'
    )
    assert _dataset_cfa_format(tmp_path) == "BayerGB12"
    # no Bayer entry anywhere -> empty string
    (dark / "metadata.json").write_text('[{"pixel_format": "Mono12"}]')
    assert _dataset_cfa_format(tmp_path) == ""
    # corrupt json is tolerated
    (dark / "metadata.json").write_text("{not json")
    assert _dataset_cfa_format(tmp_path) == ""
