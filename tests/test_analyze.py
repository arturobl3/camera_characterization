"""Pure-fit tests for the EMVA R4 Linear analysis math (no hardware).

Rows are hand-crafted roi_stats dicts: the fits only consume these
(plus the dark dict), so no synthetic images are needed for the PTC math.
"""

import numpy as np
import pytest

from camchar import analyze as A


def dark_rows():
    """Dark rows with a rising variance (500 DN16^2/s slope over 159..209)."""
    return [
        (0.001, {"mean": 165.0, "tvar": 159.0, "tf_tvar": 159.0, "tf_svar": 0.0}),
        (0.01, {"mean": 165.8, "tvar": 164.0, "tf_tvar": 164.0, "tf_svar": 0.0}),
        (0.1, {"mean": 173.0, "tvar": 209.0, "tf_tvar": 209.0, "tf_svar": 0.0}),
    ]


def dark_flat():
    """Dark rows with constant variance 159 (no dark-current variance slope)."""
    return [
        (0.001, {"mean": 165.0, "tvar": 159.0, "tf_tvar": 159.0, "tf_svar": 0.0}),
        (0.01, {"mean": 165.8, "tvar": 159.0, "tf_tvar": 159.0, "tf_svar": 0.0}),
        (0.1, {"mean": 173.0, "tvar": 159.0, "tf_tvar": 159.0, "tf_svar": 0.0}),
    ]


def dark_fit(rows=None):
    dk = A.fit_dark_stats(rows or dark_rows())
    dk["rows"] = rows or dark_rows()
    return dk


def flat_rows(
    k_slope=2.0, bias=165.0, dark_var=159.0, n=5, exp_min=0.001, exp_step=0.001
):
    """Rows with tvar = dark_var + k_slope * (mean - bias), sat_frac=0."""
    rows = []
    for i in range(n):
        exp_s = exp_min + i * exp_step
        mean = bias + 100.0 * (i + 1)
        rows.append(
            (
                exp_s,
                {
                    "mean": mean,
                    "tvar": dark_var + k_slope * (mean - bias),
                    "tf_tvar": dark_var + k_slope * (mean - bias),
                    "svar": 0.0,
                    "tf_svar": 0.0,
                    "sat_frac": 0.0,
                },
            )
        )
    return rows


class TestFitDarkStats:
    def test_bias_is_shortest_exposure_mean(self):
        dk = A.fit_dark_stats(dark_rows())
        assert dk["bias_dn"] == pytest.approx(165.0)
        # least-squares slope of mean vs t on [0.001, 0.01, 0.1]
        assert dk["dark_current_dn_per_s"] == pytest.approx(80.48, abs=0.05)
        assert dk["sigma_r_dn"] ** 2 == pytest.approx(158.72, abs=0.2)

    def test_variance_slope_matches(self):
        dk = A.fit_dark_stats(dark_rows())
        assert dk["dark_current_var_dn2_per_s"] == pytest.approx(503.0, abs=0.5)

    def test_flat_dark_gives_zero_var_slope(self):
        dk = A.fit_dark_stats(dark_flat())
        assert dk["dark_current_var_dn2_per_s"] == pytest.approx(0.0, abs=1e-9)
        assert dk["sigma_r_dn"] == pytest.approx(np.sqrt(159.0))


class TestFitFlatStats:
    def test_exact_data_recovers_k(self):
        flat = A.fit_flat_stats(flat_rows(k_slope=2.0), dark_fit(dark_flat()))
        assert flat is not None
        assert flat["K12"] == pytest.approx(8.0)
        assert flat["ptc_slope"] == pytest.approx(2.0)
        assert flat["ptc_intercept"] == 0.0  # Eq. 50 zero-intercept by construction
        assert flat["ptc_intercept_free"] == pytest.approx(0.0, abs=1e-6)
        assert flat["ptc_r2"] == pytest.approx(1.0)
        assert flat["K12_tf"] == pytest.approx(8.0)
        assert flat["Nsat"] == pytest.approx(8.0 * 4094)
        # dV aligned with rows: dV[i] = tvar[i] - dark_tvar(exp_i)
        assert flat["dV"][0] == pytest.approx(2.0 * 100.0)
        assert flat["dV"][2] == pytest.approx(2.0 * 300.0)

    def test_point_above_70pct_saturation_excluded(self):
        rows = flat_rows(n=5)
        rows[-1][1]["mean"] = 60_000.0
        rows[-1][1]["tvar"] = 159.0 + 2.0 * (60_000.0 - 165.0)
        rows[-1][1]["tf_tvar"] = rows[-1][1]["tvar"]
        flat = A.fit_flat_stats(rows, dark_fit(dark_flat()))
        assert flat is not None
        assert flat["ptc_capped"] is True
        assert flat["ptc_sat_70_dn"] == pytest.approx(0.7 * 60_000.0)
        assert flat["K12"] == pytest.approx(8.0)  # cap excluded the 60k point only

    def test_pinned_point_excluded_even_below_clip(self):
        rows = flat_rows(n=5)
        # mean 465 < 65400 clip but the majority of pixels are pinned:
        # sat_frac exceeds the EMVA 6.6 test and its variance has collapsed
        rows[2][1]["sat_frac"] = 0.01
        rows[2][1]["tvar"] = 1.0
        rows[2][1]["tf_tvar"] = 1.0
        flat = A.fit_flat_stats(rows, dark_fit(dark_flat()))
        assert flat is not None
        assert flat["dV"][2] < 0  # kept in dV (aligned), excluded from the fit
        assert flat["K12"] == pytest.approx(8.0)

    def test_negative_dv_rows_dropped(self):
        rows = flat_rows(n=5)
        # extend the top so the 70% cap keeps 4 points in the fit
        rows[-1][1]["mean"] = 1000.0
        rows[-1][1]["tvar"] = 159.0 + 2.0 * (1000.0 - 165.0)
        rows[-1][1]["tf_tvar"] = rows[-1][1]["tvar"]
        rows[0][1]["tvar"] = 1.0  # dark-subtraction noise at the lowest signal
        rows[0][1]["tf_tvar"] = 1.0
        flat = A.fit_flat_stats(rows, dark_fit(dark_flat()))
        assert flat is not None
        assert flat["dV"][0] < 0
        assert flat["K12"] == pytest.approx(8.0)

    def test_all_negative_dv_returns_none(self):
        rows = flat_rows(n=5)
        for _, s in rows:
            s["tvar"] = 1.0  # lens-cap-on scenario
        assert A.fit_flat_stats(rows, dark_fit(dark_flat())) is None

    def test_dark_var_extrapolated_outside_grid(self):
        # flat exposures (1..10 s) beyond the dark grid (max 0.1 s): the
        # endpoint slope (209-164)/(0.1-0.01) = 500 DN2/s must extend
        rows = flat_rows(n=4, k_slope=50.0, exp_min=1.0, exp_step=3.0)
        flat = A.fit_flat_stats(rows, dark_fit())
        assert flat is not None
        # dV at 1.0 s: dark tvar = 209 + 500 * (1.0 - 0.1) = 659
        assert flat["dV"][0] == pytest.approx(rows[0][1]["tvar"] - 659.0)

    def test_too_few_points_returns_none(self):
        assert A.fit_flat_stats(flat_rows(n=2), dark_fit(dark_flat())) is None


class TestTwoFrameStats:
    def test_two_frame_estimators_hand_checked(self):
        rng = np.random.default_rng(7)
        # exactly one frame pair so tf estimates equal the pair (0,1) values
        stack = rng.normal(1000.0, 10.0, (2, 8, 8)).astype(np.uint16)
        roi = (slice(None), slice(None))
        tf_t, tf_s, _, _ = A.two_frame_stats(stack, roi)
        a = stack[0].astype(np.float64)
        b = stack[1].astype(np.float64)
        mu_a, mu_b = a.mean(), b.mean()
        d = a - b
        exp_t = 0.5 * (d * d).mean() - 0.5 * (mu_a - mu_b) ** 2
        exp_s = (a * b).mean() - mu_a * mu_b
        assert tf_t == pytest.approx(exp_t, rel=1e-9)
        assert tf_s == pytest.approx(exp_s, rel=1e-9)

    def test_roi_stats_sat_frac(self):
        stack = np.zeros((3, 16, 16), dtype=np.uint16)
        stack[:, :, :] = A.SAT_MAX_DN
        r = A.roi_stats(stack, (slice(None), slice(None)))
        assert r["sat_frac"] == pytest.approx(1.0)
        assert r["tvar"] == pytest.approx(0.0)


class TestSnrTotal:
    EXTRAS = {"dsnu1288_dn": 1.0, "prnu1288_pct": 0.5}

    def test_model_matches_hand_calc(self):
        k12 = 8.0
        signal_e = 1000.0
        sigma_d_e = 5.0
        q2_e = (A.QUANT_STEP_DN16**2 / 12.0) * (k12 / 16.0) ** 2
        var = (
            sigma_d_e**2
            + (1.0 * k12 / 16.0) ** 2
            + q2_e
            + signal_e
            + (0.005 * signal_e) ** 2
        )
        expected = signal_e / np.sqrt(var)
        got = A.snr_total_model(self.EXTRAS, signal_e, sigma_d_e, k12)
        assert got == pytest.approx(expected)

    def test_measured_matches_hand_calc(self):
        k12 = 8.0
        bias = 165.0
        means = np.array([165.0 + 500.0, 165.0 + 2000.0])
        tvars = np.array([400.0, 800.0])
        s2 = 1.0**2 + (0.005 * (means - bias)) ** 2
        signal_e = (means - bias) * k12 / 16.0
        expected = signal_e / np.sqrt((tvars + s2) * (k12 / 16.0) ** 2)
        got = A.snr_total_measured(self.EXTRAS, means, bias, tvars, k12)
        np.testing.assert_allclose(got, expected)
