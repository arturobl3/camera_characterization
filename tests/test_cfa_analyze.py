"""Pure tests for the per-CFA-channel analysis (no hardware)."""

import csv

import numpy as np

from camchar import cfa_analyze as C

# ROI well inside the 16x16 frames
_ROI = (slice(2, 6), slice(4, 8))


def _stack(mean_by_phase, n=4, h=16, w=16):
    """(n, h, w) uint16 stack with per-phase constant values (BGGR labels)."""
    out = np.empty((n, h, w), dtype=np.uint16)
    for k in range(n):
        for (i, j), v in mean_by_phase.items():
            out[k, i::2, j::2] = v
    return out


def test_channel_slicing_matches_manual_phases():
    # distinct pedestal per phase; dark rows must report exactly these means
    means = {(0, 0): 200, (0, 1): 300, (1, 0): 400, (1, 1): 500}
    seq = [(float(e), _stack(means)) for e in (0.001, 0.002)]
    res = C._analyze_channel("B", 0, 0, seq, [], _ROI)
    assert [s["mean"] for _, s in res["dark_rows"]] == [200.0, 200.0]
    assert res["status"] == "no flat data"
    # a different phase on the same stacks sees its own value
    res_r = C._analyze_channel("R", 1, 1, seq, [], _ROI)
    assert [s["mean"] for _, s in res_r["dark_rows"]] == [500.0, 500.0]


def test_status_no_dark_data():
    res = C._analyze_channel("R", 0, 0, [], [(0.01, _stack({}))], _ROI)
    assert res["status"] == "no dark data"
    assert res["flat"] is None


def test_status_skipped_without_usable_flats():
    dark = [(0.001, _stack({(0, 0): 200}))] * 3
    # flat rows with <3 usable points (all saturated above clip)
    flat = [(e, _stack({(0, 0): 65504})) for e in (0.01, 0.02, 0.04, 0.08)]
    res = C._analyze_channel("B", 0, 0, dark, flat, _ROI)
    assert res["status"].startswith("skipped")
    assert res["flat"] is None


def test_run_labels_follow_bayer_format():
    """run()'s phase mapping labels BGGR channels B,G1,G2,R with colors."""
    # distinct pedestals prove the mapping
    means = {(0, 0): 100, (0, 1): 200, (1, 0): 300, (1, 1): 400}
    dark = [(0.001, _stack(means))]
    labels = C._cfa_labels("BayerBG12")
    phases = list(zip(labels, (0, 0, 1, 1), (0, 1, 0, 1)))
    results = [
        C._analyze_channel(label, i, j, dark, [], _ROI) for label, i, j in phases
    ]
    assert [r["label"] for r in results] == ["B", "G1", "G2", "R"]
    by_label = {r["label"]: r for r in results}
    assert by_label["B"]["dark_rows"][0][1]["mean"] == 100.0
    assert by_label["G1"]["dark_rows"][0][1]["mean"] == 200.0
    assert by_label["G2"]["dark_rows"][0][1]["mean"] == 300.0
    assert by_label["R"]["dark_rows"][0][1]["mean"] == 400.0
    assert by_label["R"]["plot_color"] == "tab:red"
    assert by_label["B"]["plot_color"] == "tab:blue"


def test_write_csv_columns_and_rows(tmp_path):
    results = [
        {
            "label": "B",
            "status": "no flat data",
            "dark": {
                "bias_dn": 83.5,
                "sigma_r_dn": 13.7,
                "sigma_r_median_dn": 13.8,
                "dark_current_dn_per_s": 1.2345,
                "dark_current_var_dn2_per_s": 5.0,
                "dark_current_var_dn2_per_s_nf": 5.1,
            },
            "flat": None,
            "extras": None,
        },
        {"label": "R", "status": "skipped (0 usable flat pts)", "dark": None},
    ]
    path = tmp_path / "cfa_parameters.csv"
    C._write_csv(results, path)
    rows = list(csv.reader(open(path)))
    assert rows[0][0] == "channel"
    assert "K12_e_per_DN12" in rows[0]
    assert "dsnu1288_dn16" in rows[0]
    assert len(rows) == 3
    b_row = dict(zip(rows[0], rows[1]))
    assert b_row["channel"] == "B"
    assert b_row["bias_dn16"] == "83.50"
    r_row = dict(zip(rows[0], rows[2]))
    assert r_row["status"] == "skipped (0 usable flat pts)"
    assert r_row["bias_dn16"] == ""
