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


def invoke(monkeypatch, args, backend):
    monkeypatch.setattr(cli, "get_backend", lambda name: backend())
    return runner.invoke(cli.app, args)


def test_stability_unstable(monkeypatch):
    r = invoke(
        monkeypatch, ["source-stability-check", "--vendor", "fake"], UnstableBackend
    )
    assert r.exit_code == 1
    assert "EXCEEDS THRESHOLD" in r.output


def test_stability_stable(monkeypatch):
    r = invoke(
        monkeypatch, ["source-stability-check", "--vendor", "fake"], StableBackend
    )
    assert r.exit_code == 0
    assert "stable at all exposures" in r.output


def test_warmup_stable(monkeypatch):
    monkeypatch.setattr(cli, "WARMUP_STABLE_WINDOW_S", 0.2)
    monkeypatch.setattr(cli, "WARMUP_STABLE_TOL_C", 0.3)
    monkeypatch.setattr(cli, "WARMUP_STOP_DELAY_S", 0.2)
    monkeypatch.setattr(cli, "WARMUP_PRINT_INTERVAL_S", 0.1)
    r = invoke(monkeypatch, ["warmup-sensor", "--vendor", "fake"], StableBackend)
    assert r.exit_code == 0
    assert "warmup complete" in r.output


def test_warmup_interrupted(monkeypatch):
    r = invoke(monkeypatch, ["warmup-sensor", "--vendor", "fake"], InterruptBackend)
    assert r.exit_code == 130


def test_warmup_aborted(monkeypatch):
    monkeypatch.setattr(cli, "WARMUP_MAX_CONSECUTIVE_FAILS", 2)
    r = invoke(monkeypatch, ["warmup-sensor", "--vendor", "fake"], FailBackend)
    assert r.exit_code == 1


def test_bad_exposures(monkeypatch):
    r = invoke(
        monkeypatch,
        ["source-stability-check", "--vendor", "fake", "--exposures", "0.001,abc"],
        StableBackend,
    )
    assert r.exit_code == 2


def test_missing_vendor(monkeypatch):
    r = invoke(monkeypatch, ["source-stability-check"], StableBackend)
    assert r.exit_code == 2
    assert "Missing option" in r.output
