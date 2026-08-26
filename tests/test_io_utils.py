"""Pure-function tests for io_utils naming/metadata conventions."""

import numpy as np

from camchar import analyze as A
from camchar.io_utils import camera_dir_name, central_roi, save_sequence, stem_for


def _stack(frame_means, h=4, w=4):
    return np.stack([np.full((h, w), v, dtype=np.uint16) for v in frame_means])


class TestSaveSequenceGuard:
    def test_warns_on_frame_mean_step(self, tmp_path, capsys):
        info = {"vendor": "thorlabs", "model": "LP126CU", "sensor": ""}
        save_sequence(
            tmp_path, "dark", 0.569, 0, _stack([1600, 1600, 1088, 1088]), info
        )
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "black-level" in out
        assert "512.00 DN16" in out  # spread 1600-1088 = 512

    def test_no_warning_when_frame_means_stable(self, tmp_path, capsys):
        info = {"vendor": "thorlabs", "model": "LP126CU", "sensor": ""}
        save_sequence(tmp_path, "dark", 0.1, 0, _stack([1600] * 5), info)
        out = capsys.readouterr().out
        assert "WARNING" not in out
        assert "saved" in out


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


class TestCentralRoi:
    def test_imx174_frame(self):
        assert central_roi(1920, 1200) == (300, 900, 480, 1440)

    def test_square_ace2_frame(self):
        assert central_roi(3536, 3536) == (884, 2652, 884, 2652)

    def test_large_thorlabs_frame(self):
        assert central_roi(4096, 3000) == (750, 2250, 1024, 3072)

    def test_odd_dimensions_even_sides(self):
        r = central_roi(1001, 601)
        assert r == (150, 450, 250, 750)
        assert (r[1] - r[0]) % 2 == 0
        assert (r[3] - r[2]) % 2 == 0

    def test_tiny_frame_clamped_to_min_side(self):
        assert central_roi(4, 4) == (1, 3, 1, 3)
        assert central_roi(2, 2) == (0, 2, 0, 2)

    def test_custom_fraction(self):
        assert central_roi(1000, 1000, 0.25) == (375, 625, 375, 625)

    def test_window_inside_frame(self):
        for w, h in ((1920, 1200), (3536, 3536), (4096, 3000), (7, 9), (2, 2)):
            r0, r1, c0, c1 = central_roi(w, h)
            assert 0 <= r0 < r1 <= h
            assert 0 <= c0 < c1 <= w
