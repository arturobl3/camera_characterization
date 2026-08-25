"""CLI exit-code tests: Typer CliRunner with mocked backends (no hardware)."""

import numpy as np
from typer.testing import CliRunner

import camchar.cli as cli

runner = CliRunner()


class FakeBackend:
    def open(self):
        return {
            "vendor": "fake",
            "model": "FakeCam",
            "serial": "0",
            "sensor": "FAKE",
            "pixel_size_um": 1.0,
            "bit_depth": 12,
            "width": 4,
            "height": 4,
            "usb3": False,
        }

    def configure(self, gain=0):
        return {}

    def sensor_temp_c(self):
        return 25.0

    def close(self):
        pass


def stack_from(means, n):
    return np.stack([np.full((4, 4), v, dtype=np.uint16) for v in means[:n]])


class UnstableBackend(FakeBackend):
    def snap(self, exp, n):
        m = {
            1e-5: np.array([180.0, 180.2, 180.4, 180.1]),  # ~+0.11%
            1e-4: np.array([1552.33, 1553.11, 1554.19, 1552.02]),
            0.001: np.array([1552.33, 1553.11, 1554.19, 1552.02]),  # +0.12%
            0.01: np.array([12716.0, 12717.2, 12716.8, 12717.5]),
            0.1: np.array([60000.0, 60002.0, 59999.0, 60001.0]),
            1.0: np.array([60000.0, 60002.0, 59999.0, 60001.0]),
        }[round(exp, 9)]
        return stack_from(m, n)


class StableBackend(FakeBackend):
    def snap(self, exp, n):
        m = {
            1e-5: np.array([180.0, 180.02, 180.01, 180.0]),
            1e-4: np.array([1552.33, 1552.50, 1552.20, 1552.60]),
            0.001: np.array([1552.33, 1552.50, 1552.20, 1552.60]),
            0.01: np.array([12716.0, 12717.2, 12716.8, 12717.5]),
            0.1: np.array([60000.0, 60002.0, 59999.0, 60001.0]),
            1.0: np.array([60000.0, 60002.0, 59999.0, 60001.0]),
        }[round(exp, 9)]
        return stack_from(m, n)


class InterruptBackend(FakeBackend):
    def snap(self, exp, n):
        raise KeyboardInterrupt


class FailBackend(FakeBackend):
    def snap(self, exp, n):
        raise RuntimeError("simulated USB hiccup")


class NoTempBackend(FakeBackend):
    has_temperature = False


class SatSweepBackend(FakeBackend):
    """check-saturation backend: 1 frame per exposure from a ms->mean map.

    mean values >= 65504 count as saturated frames (every pixel clipped).
    """

    means_ms = {}

    def snap(self, exp, n):
        ms = round(exp * 1000.0, 6)
        v = self.means_ms[ms]
        return np.full((1, 8, 8), v, dtype=np.uint16)


class SatIdealBackend(SatSweepBackend):
    # first saturation exactly at exposures[-3] (the 40 ms point)
    means_ms = {5.0: 1000, 10.0: 2000, 40.0: 65504}


class SatTooBrightBackend(SatSweepBackend):
    # saturates immediately -> reduce intensity
    means_ms = {10.0: 65504}


class SatDimBackend(FakeBackend):
    # linear low signal, nothing saturates -> extrapolated increase factor
    def snap(self, exp, n):
        v = int(round(exp * 100000.0))  # 100 DN16 per ms
        return np.full((1, 8, 8), v, dtype=np.uint16)


class SatZeroSignalBackend(FakeBackend):
    def snap(self, exp, n):
        return np.zeros((1, 8, 8), dtype=np.uint16)


def invoke(monkeypatch, args, backend):
    monkeypatch.setattr(cli, "get_backend", lambda name: backend())
    return runner.invoke(cli.app, args)


def test_stability_unstable(monkeypatch):
    r = invoke(
        monkeypatch, ["source-stability-check", "--vendor", "basler"], UnstableBackend
    )
    assert r.exit_code == 1
    assert "EXCEEDS THRESHOLD" in r.output


def test_stability_stable(monkeypatch):
    r = invoke(
        monkeypatch, ["source-stability-check", "--vendor", "basler"], StableBackend
    )
    assert r.exit_code == 0
    assert "stable at all exposures" in r.output


def test_warmup_stable(monkeypatch):
    monkeypatch.setattr(cli, "WARMUP_STABLE_WINDOW_S", 0.2)
    monkeypatch.setattr(cli, "WARMUP_STABLE_TOL_C", 0.3)
    monkeypatch.setattr(cli, "WARMUP_STOP_DELAY_S", 0.2)
    monkeypatch.setattr(cli, "WARMUP_PRINT_INTERVAL_S", 0.1)
    r = invoke(monkeypatch, ["warmup-sensor", "--vendor", "basler"], StableBackend)
    assert r.exit_code == 0
    assert "warmup complete" in r.output


def test_warmup_interrupted(monkeypatch):
    r = invoke(monkeypatch, ["warmup-sensor", "--vendor", "basler"], InterruptBackend)
    assert r.exit_code == 130


def test_warmup_aborted(monkeypatch):
    monkeypatch.setattr(cli, "WARMUP_MAX_CONSECUTIVE_FAILS", 2)
    r = invoke(monkeypatch, ["warmup-sensor", "--vendor", "basler"], FailBackend)
    assert r.exit_code == 1


def test_warmup_no_temperature_backend(monkeypatch):
    r = invoke(monkeypatch, ["warmup-sensor", "--vendor", "basler"], NoTempBackend)
    assert r.exit_code == 1
    assert "no sensor temperature" in r.output


def test_bad_exposures(monkeypatch):
    r = invoke(
        monkeypatch,
        ["source-stability-check", "--vendor", "basler", "--exposures", "0.001,abc"],
        StableBackend,
    )
    assert r.exit_code == 2


def test_missing_vendor(monkeypatch):
    r = invoke(monkeypatch, ["source-stability-check"], StableBackend)
    assert r.exit_code == 2
    assert "Missing option" in r.output


def test_invalid_vendor(monkeypatch):
    """Unknown vendor exits 2 at option validation, before any backend lookup."""

    def _fail(name):  # must never be reached for an unregistered vendor name
        raise AssertionError("get_backend called for invalid vendor")

    monkeypatch.setattr(cli, "get_backend", _fail)
    r = runner.invoke(cli.app, ["source-stability-check", "--vendor", "fake"])
    assert r.exit_code == 2
    assert "not one of" in r.output


def test_saturation_ideal(monkeypatch):
    r = invoke(
        monkeypatch,
        [
            "check-saturation",
            "--vendor",
            "basler",
            "--exposures",
            "5,10,40,80,160",
        ],
        SatIdealBackend,
    )
    assert r.exit_code == 0
    assert "ideal" in r.output


def test_saturation_too_bright(monkeypatch):
    r = invoke(
        monkeypatch,
        [
            "check-saturation",
            "--vendor",
            "basler",
            "--exposures",
            "10,20,40,80,160",
        ],
        SatTooBrightBackend,
    )
    assert r.exit_code == 1
    assert "reduce intensity" in r.output
    assert "x4.00" in r.output


def test_saturation_too_dim(monkeypatch):
    # slope 100 DN16/ms -> predicted onset at 65000/100 = 650 ms;
    # target (3rd of 5) = 40 ms -> increase by x16.25
    r = invoke(
        monkeypatch,
        [
            "check-saturation",
            "--vendor",
            "basler",
            "--exposures",
            "5,10,40,80,160",
        ],
        SatDimBackend,
    )
    assert r.exit_code == 1
    assert "increase intensity" in r.output
    assert "x16.25" in r.output


def test_saturation_no_signal(monkeypatch):
    r = invoke(
        monkeypatch,
        ["check-saturation", "--vendor", "basler", "--exposures", "1,2,4"],
        SatZeroSignalBackend,
    )
    assert r.exit_code == 1
    assert "no usable signal" in r.output


def test_saturation_descending_exposures_rejected(monkeypatch):
    r = invoke(
        monkeypatch,
        ["check-saturation", "--vendor", "basler", "--exposures", "160,80,40"],
        SatIdealBackend,
    )
    assert r.exit_code == 2
    assert "ascending" in r.output
