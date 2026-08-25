"""Pure-function tests for the thorlabs backend helpers (no hardware)."""

from camchar.backends.thorlabs import _pixel_format


def test_pixel_format_mono():
    # SENSOR_TYPE.MONOCHROME == 0
    assert _pixel_format(0, None, 12) == "Mono12"
    assert _pixel_format(0, 2, 12) == "Mono12"  # phase ignored on mono


def test_pixel_format_bayer_layouts():
    # FILTER_ARRAY_PHASE: 0=RED, 1=BLUE, 2=G_LEFT_OF_RED, 3=G_LEFT_OF_BLUE
    assert _pixel_format(1, 0, 12) == "BayerRG12"
    assert _pixel_format(1, 1, 12) == "BayerBG12"  # LP126CU (origin blue)
    assert _pixel_format(1, 2, 12) == "BayerGR12"
    assert _pixel_format(1, 3, 12) == "BayerGB12"


def test_pixel_format_bayer_unknown_phase():
    # layout unknown -> still flagged as Bayer; labels fall back to RGGB
    assert _pixel_format(1, None, 12) == "Bayer12"
    assert _pixel_format(1, 99, 8) == "Bayer8"


def test_pixel_format_bit_depth_passthrough():
    assert _pixel_format(1, 1, 10) == "BayerBG10"
    assert _pixel_format(0, None, 8) == "Mono8"
