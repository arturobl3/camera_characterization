"""Pure-function tests for io_utils naming/metadata conventions."""

import numpy as np

from camchar import analyze as A
from camchar.io_utils import camera_dir_name, save_sequence, stem_for


class TestStemFor:
    def test_integer_ms(self):
        assert stem_for("dark", 0.100, 0) == "dark_00100ms_g0"

    def test_fractional_ms_uses_us(self):
        assert stem_for("flat", 0.0045, 0) == "flat_04500us_g0"

    def test_sub_ms(self):
        assert stem_for("dark", 2.1e-5, 0) == "dark_00021us_g0"

    def test_float_gain_formats_compact(self):
        assert stem_for("dark", 0.1, 0.0) == "dark_00100ms_g0"
        assert stem_for("dark", 0.1, 5.5) == "dark_00100ms_g5.5"

    def test_metadata_round_trip(self, tmp_path):
        """save_sequence + load_sequence agree through the naming convention."""
        info = {
            "vendor": "basler",
            "model": "acA1920-155um",
            "serial": "1",
            "sensor": "IMX174",
            "pixel_size_um": 5.86,
            "bit_depth": 12,
            "width": 4,
            "height": 4,
            "usb3": True,
        }
        cases = [(0.001, 0.0), (2.1e-5, 0.0), (0.0045, 5.5)]
        cam_dir = tmp_path / "camdir"
        for exposure_s, gain in cases:
            stack = np.zeros((3, 4, 4), dtype=np.uint16)
            save_sequence(cam_dir / "dark", "dark", exposure_s, gain, stack, info)
            stem = stem_for("dark", exposure_s, gain)
            assert (cam_dir / "dark" / f"{stem}.npy").exists()

        # append-only metadata: a duplicate (exposure, gain) entry from a
        # later run is deduped by load_sequence; npy files are ground truth
        save_sequence(
            cam_dir / "dark",
            "dark",
            0.001,
            0.0,
            np.zeros((3, 4, 4), dtype=np.uint16),
            info,
        )
        seq = A.load_sequence(cam_dir, "dark")
        assert len(seq) == 3
        assert [e for e, _ in seq] == sorted(e for e, _ in cases)


class TestCameraDirName:
    def test_full(self):
        info = {"vendor": "basler", "model": "acA1920-155um", "sensor": "IMX174"}
        assert camera_dir_name(info) == "basler_acA1920-155um_(IMX174)"

    def test_spaces_become_dashes(self):
        info = {"vendor": "playerone", "model": "Apollo-M", "sensor": "IMX174"}
        assert camera_dir_name(info) == "playerone_Apollo-M_(IMX174)"

    def test_missing_sensor_omitted(self):
        info = {"vendor": "x", "model": "y"}
        assert camera_dir_name(info) == "x_y"

    def test_missing_vendor_model_never_raises(self):
        assert camera_dir_name({"sensor": "IMX174"}) == "_(IMX174)"
