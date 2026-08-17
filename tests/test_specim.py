"""Unit tests for the SPECIM IQ ENVI loader (synthetic BIL cubes, no hardware).

Covers the mmap ROI reader (byte-identical to the on-disk BIL layout), the
DN12<<4 shift, cube filtering by digit stems, and the header cross-checks.
"""

import numpy as np

from camchar import specim

WL = [100.0, 200.0, 300.0, 400.0]


def write_cube(cap_dir, stem, lines, samples, bands, data, tint_ms=1, wl=WL):
    """One ENVI cube: <stem>.hdr + <stem>.raw (BIL uint16), like the IQ export."""
    raw = cap_dir / f"{stem}.raw"
    data.astype(np.uint16).tofile(raw)
    wl_text = ",\n".join(f"  {w}" for w in wl)
    hdr = (
        "ENVI\n"
        f"samples = {samples}\n"
        f"lines = {lines}\n"
        f"bands = {bands}\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 12\n"
        "interleave = bil\n"
        "byte order = 0\n"
        f"tint = {tint_ms}\n"
        "wavelength = {\n"
        f"{wl_text}}}\n"
    )
    (cap_dir / f"{stem}.hdr").write_text(hdr)
    return raw


def make_layout(tmp_path, exp_ms="1 ms", stem="001", tint_ms=1):
    """<root>/dark frames/<exp_ms>/<stem>/capture/ with one cube."""
    root = tmp_path / "specim"
    cap_dir = root / "dark frames" / exp_ms / stem / "capture"
    cap_dir.mkdir(parents=True)
    return root, cap_dir


def bil_cube(lines=8, samples=10, bands=4):
    """Deterministic cube in BIL on-disk order (lines, bands, samples)."""
    return np.arange(lines * samples * bands, dtype=np.uint16).reshape(
        lines, bands, samples
    )


class TestReadRoiBil:
    def test_roi_values(self, tmp_path):
        cube = bil_cube()
        raw = tmp_path / "c.raw"
        cube.tofile(raw)
        out = specim.read_roi_bil(raw, 8, 10, 4, 2, 6, 3, 7)
        expected = np.ascontiguousarray(cube[2:6, :, 3:7].transpose(0, 2, 1))
        assert out.shape == (4, 4, 4)
        assert out.dtype == np.uint16
        assert out.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(out, expected)

    def test_header_offset(self, tmp_path):
        cube = bil_cube()
        raw = tmp_path / "c.raw"
        with open(raw, "wb") as f:
            f.write(b"\x00" * 64)
            cube.tofile(f)
        out = specim.read_roi_bil(raw, 8, 10, 4, 0, 8, 0, 10, offset=64)
        expected = np.ascontiguousarray(cube.transpose(0, 2, 1))
        np.testing.assert_array_equal(out, expected)


class TestIterCubes:
    def test_only_digit_stems(self, tmp_path):
        root, cap_dir = make_layout(tmp_path)
        cube = bil_cube()
        write_cube(cap_dir, "001", 8, 10, 4, cube)
        write_cube(cap_dir, "DARKREF_001", 8, 10, 4, cube)
        write_cube(cap_dir, "WHITEREF_001", 8, 10, 4, cube)
        hdrs = specim.iter_cubes(root / "dark frames" / "1 ms")
        assert [h.stem for h in hdrs] == ["001"]


class TestListExposures:
    def test_sorted_and_filtered(self, tmp_path):
        root, _ = make_layout(tmp_path, exp_ms="1 ms")
        for name in ("10 ms", "2 ms", "not-an-exposure", "1 ms"):
            (root / "dark frames" / name).mkdir(exist_ok=True)
        exps = specim.list_exposures(root / "dark frames")
        assert [e for e, _ in exps] == [0.001, 0.002, 0.01]


class TestLoadExposureStack:
    def test_stack_values_and_wavelengths(self, tmp_path):
        root, cap_dir = make_layout(tmp_path, exp_ms="1 ms", tint_ms=1)
        cube = bil_cube() % 4096
        write_cube(cap_dir, "001", 8, 10, 4, cube, tint_ms=1)
        exp_dir = root / "dark frames" / "1 ms"
        stack, wl = specim.load_exposure_stack(exp_dir, 0.001, 2, 6, 3, 7)
        expected = (np.ascontiguousarray(cube[2:6, :, 3:7].transpose(0, 2, 1))) << 4
        assert stack.shape == (1, 4, 4, 4)
        assert stack.dtype == np.uint16
        np.testing.assert_array_equal(stack[0], expected)
        np.testing.assert_array_equal(wl, np.array(WL))

    def test_multiple_cubes_order(self, tmp_path):
        root, cap_dir = make_layout(tmp_path, exp_ms="1 ms", tint_ms=1)
        cube = bil_cube() % 4096
        write_cube(cap_dir, "001", 8, 10, 4, cube, tint_ms=1)
        write_cube(cap_dir, "010", 8, 10, 4, cube + 1, tint_ms=1)
        stack, _ = specim.load_exposure_stack(
            root / "dark frames" / "1 ms", 0.001, 0, 8, 0, 10
        )
        assert stack.shape == (2, 8, 10, 4)
        np.testing.assert_array_equal(stack[0], cube.transpose(0, 2, 1) << 4)
        np.testing.assert_array_equal(stack[1], (cube + 1).transpose(0, 2, 1) << 4)

    def test_tint_mismatch_warns(self, tmp_path, capsys):
        root, cap_dir = make_layout(tmp_path, exp_ms="1 ms", tint_ms=5)
        write_cube(cap_dir, "001", 8, 10, 4, bil_cube() % 4096, tint_ms=5)
        stack, _ = specim.load_exposure_stack(
            root / "dark frames" / "1 ms", 0.001, 0, 8, 0, 10
        )
        assert stack is not None
        assert "hdr tint 5 ms != folder 1 ms" in capsys.readouterr().out

    def test_over_12bit_warns(self, tmp_path, capsys):
        root, cap_dir = make_layout(tmp_path, exp_ms="1 ms", tint_ms=1)
        cube = bil_cube() % 4096
        cube[0, 0, 0] = 5000
        write_cube(cap_dir, "001", 8, 10, 4, cube, tint_ms=1)
        stack, _ = specim.load_exposure_stack(
            root / "dark frames" / "1 ms", 0.001, 0, 8, 0, 10
        )
        assert "max DN 5000 > 4095" in capsys.readouterr().out
        assert stack[0, 0, 0, 0] == (5000 << 4) & 0xFFFF  # wraps, as warned

    def test_no_cubes_returns_none(self, tmp_path):
        root, cap_dir = make_layout(tmp_path, exp_ms="1 ms", tint_ms=1)
        write_cube(cap_dir, "DARKREF_001", 8, 10, 4, bil_cube(), tint_ms=1)
        out = specim.load_exposure_stack(
            root / "dark frames" / "1 ms", 0.001, 0, 8, 0, 10
        )
        assert out == (None, None)
