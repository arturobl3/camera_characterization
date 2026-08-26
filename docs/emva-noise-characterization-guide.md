# Camera Noise Characterization — EMVA 1288 Step-by-Step Guide

Practical, field-tested walkthrough of the photon-transfer / EMVA 1288 methodology,
written from actual measurement campaigns (IMX174, both a Player One Apollo-M and a
Basler acA1920-155um; plus a SPECIM IQ hyperspectral). Every step below is the
distilled, corrected version after we made (and fixed) the classic errors. All
equation numbers refer to **EMVA 1288 Release 4.0 Linear** (`docs/EMVA1288Linear_4.0Release.pdf`
in this repo) unless stated otherwise.

Use this together with the `camchar` CLI (see `README.md` / `AGENTS.md`): each step
maps to a `camchar` command or to numbers printed by `camchar analyze`.

---

## Parameter → step cheat sheet

| Parameter | Symbol | Step | Primary source |
|---|---|---|---|
| Bias offset | µd0 | 2 | intercept of µd(t) |
| Dark current | i_d | 2 | slope of µd(t) (mean-based, trustworthy) |
| Read noise | σr (EMVA: σd0) | 2 | √σ²y.dark at short exposure, t→0 |
| System gain | K | 3 | slope of photon transfer curve (plateau region) |
| Saturation capacity | µe.sat / Nsat | 4 | variance knee (EMVA) or hard clip × K |
| Quantum efficiency | η(λ) | 5 | sensitivity curve slope / K (needs radiometry) |
| DSNU | DSNU1288 | 6 | spatial std of dark offset pattern |
| PRNU | PRNU1288 | 6 | spatial std of light-induced pattern / signal |
| SNR, DR, threshold | — | 7 | combine 2–6 |

---

## Step 0 — Data recording

**Goal.** For a sweep of exposure values, record stacks of *dark frames* (lens cap
on, completely dark) and *flat frames* (camera facing a uniform, stable light
source). Darks yield dark current, read noise, DSNU. Flats yield the photon
transfer curve → gain K, saturation, PRNU (and QE with a calibrated light).

**Exposure sweep.** The EMVA guidance on point counts applies to the **flats**
(the point distribution matters there): at least 9–15 well-chosen levels for lab
work, ≥50 for a full datasheet. Include:

- **Flats** (10–15 levels): 2–3 points below ~10 % of saturation (read-noise
  dominated), most points spread over the linear range, 3–4 points in the top
  20 %, and **1–2 clearly saturated points** (the sweep must *cross* the knee).
- **Darks** (two roles): (a) a dark pair at each flat exposure time, for
  subtraction; (b) a dedicated longer dark-only sweep (1 ms → seconds) to
  resolve the dark-current slope — this one is never skippable. If dark current
  is provably negligible over the flat range, a single short dark can serve all
  flat levels (not EMVA-blessed but defensible).

**Frames per level:** 2 minimum (EMVA two-image method); 20 is a good lab default
(unbiased variance, drift diagnostics, PRNU stacks). Long exposures: fewer frames
are acceptable — the mean is what matters there.

**Light source.** A stable uniform source (halogen + diffuser, ≥ 30 min warm-up)
is the gold standard. PWM sources are *risky*, not "usable with a workaround":
the two-frame estimator corrects only spatially-uniform common-mode fluctuation
and cannot detect its own failure. Qualify any source: frame-mean stability
< ~0.1 %, and the N-frame vs two-frame estimates must agree to < ~1 %.

**Temperature.** Warm up to thermal equilibrium (drift < ~0.3 °C over 2 min) and
record sensor temperature with every sequence. Dark current doubles every
~6–7 °C — report it **with its temperature**, and at this operating point keep
flat exposures short enough that dark-current shot noise stays small. K, σr,
Nsat, PRNU are effectively temperature-independent; consistency matters for
comparability, and a dedicated i_d(T) sweep is needed for a temperature-aware
simulator.

**Firmware auto-black-level while warming (LP126CU lesson, Aug 2026).** Some
camera firmwares re-zero their digital black pedestal mid-sweep — observed on the
Thorlabs LP126CU: the sensor's internal black-level controller (no SDK control)
steps the digital pedestal down in **integer DN12 steps** (measured −2, later
±1 more) when the *accumulated dark signal* (dark current × exposure) exceeds
~1–2 DN12 during an escalating dark sweep once the sensor is warm (dark current
≳ 1 DN12/s). Setting the SDK `black_level` register higher does NOT stop it
(verified: the same −2 DN12 step fires at BL=100); constant-exposure runs settle
and are stable, escalating sweeps drift above the controller's target and step.
A dark curve that rises, then **collapses below bias** (a dark mean below the
short-exposure pedestal is physically impossible) is the tell-tale signature.
Protocol:

- **Cap the dark sweep below the step threshold**: at warm lab conditions that
  is ~300–400 ms (steps fired at 390–569 ms in every observed session; dark×t ≈
  1 DN12 there). The analysis extrapolates dark variance linearly past the dark
  grid, so a capped sweep is fully valid — a long dark tail is not worth
  corrupting the curve.
- **Drop, don't fit through, any flagged point**: `camchar` warns loudly at
  acquisition (per-frame "±X across frames" quality line; `save_sequence` guard
  at >4 DN16 spread) and `camchar analyze` flags the affected dark rows
  (`frame_mean_spread`) so recorded datasets are self-diagnosing.
- **Heat-soaking does NOT prevent it**: `warmup-sensor` reaches thermal steady
  state but the controller still steps during the following escalating sweep —
  verified live (an engaged, stable controller at 1 s still stepped the next
  sweep at 569 ms).

---

## Step 1 — Per-pixel temporal statistics

**Goal.** Compress each stack (N frames × H × W) into one row of three numbers
per exposure: **µy** (mean signal), **σ²y** (temporal variance), **s²y** (spatial
variance). All later steps consume only these rows — one per exposure, separate
for dark and flat.

**How.** Compute *per pixel first, then average over pixels*:

- For each pixel: µy(x,y) = mean over frames; σ²y(x,y) = variance over frames.
- Over the ROI (a few thousand pixels; only the ROI needs uniform illumination):
  - µy = mean of pixel means
  - σ²y = mean of pixel temporal variances → shot + read + dark-current noise
  - s²y = variance of pixel means → fixed pattern (DSNU + PRNU + illumination

Why the split is safe: fixed patterns are identical in every frame → zero
temporal variance contribution (the temporal estimator cannot see them); random
noise is frame-independent → averages out of the pixel means (spatial estimator).
Doing it the other way (variance of whole frames) mixes fixed pattern into
"noise" — the failure mode that made a PWM screen look 200× noisier than shot
noise.

**Two equivalent estimators** (same three outputs, two nuances):

- **N-frame stack**: σ²y = per-pixel variance over N frames with **ddof=1**
  (EMVA Eq. 44/65). The population estimator (ddof=0) is biased low by
  (N−1)/N — 5 % at N=20 — which silently shifts K (high) and σr (low). s²y needs
  the −σ²y/N correction. Includes any common-mode fluctuation.
- **Two-frame difference** (EMVA R4, §6.5/6.6): σ²y = mean((yA−yB)²)/2 −
  (µA−µB)²/2 (Eq. 18, common-mode corrected); s²y from pair covariance (Eq. 32,
  exact, no correction). Minimum 2 frames; precision from pixel averaging
  (~0.15 % at a megapixel); requires stationarity between frames.

**Cross-check invariant.** The N-frame and two-frame estimates must agree to
< ~1 %. A systematic gap = estimator bias (the ddof bug) or source non-stationarity
— investigate, don't average. The N-frame route additionally exposes drift/flicker
in the frame-mean series that the two-frame route is blind to.

**Units.** RAW16 here stores 12-bit values left-shifted ×4 (DN16 = DN12 << 4):
signals ×16, variances ×256. Do all math in one scale and convert consistently.

---

## Step 2 — Dark analysis (bias, dark current, read noise)

**Goal.** From the dark rows (µy.dark, σ²y.dark vs t): **bias offset µd0**,
**dark current i_d**, **read noise** (EMVA "temporal dark noise" σd0), and DSNU
(deferred to Step 6 for the full treatment).

**Model** (EMVA §3):
```
E[µ].dark(t) = µd0 + i_d·t                              (Eq. 29, mean)
σ²d(t) = σ²d0 + (i_d·t)/K                              (Eq. 30, variance)
```
The second is just Eq. 30 in DN units: the Poisson identity (variance = mean)
gives σ²d,e = µI_therm = µI·t in e⁻; divide by K² to get DN² with i_d = µI/K
(DN/s): σ²d,DN = (µI·t)/K² = (i_d·t)/K. The e⁻ is a dimensionless count in all
noise formulas.

**Extraction.**

1. **Bias** — intercept of µd(t) (or mean at the shortest exposure). Ours:
   164.9 DN16 = 10.3 DN12 (it's a real offset, not zeros).
2. **Dark current** — slope of µd(t), converted to e⁻/s with K (from Step 3):
   12.15 DN16/s ≈ 6.4 e⁻/s @ 29.5 °C (IMX174). Always report temperature.
3. **Read noise σd0** — from the *flat* part of σ²d(t) (short exposures:
   read-dominated): median of short-exposure σ²d → √σ² ≈ 6.6 e⁻ (Basler's
   certified 6.8 e⁻). In DN it needs no K; in e⁻ it does — so σd0 in DN is the
   Step-2 output; the e⁻ conversion waits for Step 3.
4. **DSNU** — spatial variance of the dark offset pattern via two-frame
   covariance (Eq. 32; clean, no hot-pixel games). Ours ~1.1 DN16 ≈ 0.6 e⁻
   (Basler full-frame: 1.9 e⁻).
5. **Cross-check** — the slope of σ²d(t) at long exposures must equal i_d/K
   (mean-vs-variance dark current agree within temp drift; ours 26 vs 23 DN²/s).

**Checks.**

- σ²d(t) flat at short exposures → read-noise dominance; a rising tail = dark
  shot, not a defect.
- Truly dark: stray light masquerades as dark current (slope of µd(t) jumps).
- Quantization floor: σ²d ≳ 0.24 DN² (EMVA minimum; native DN — see Units
  appendix). If not, use the σ²q = 1/12 DN² correction; if σ²d < 0.24 the true
  σd canNOT be resolved (see Quantization appendix, Eq. 54).
- Record temperature: drift of 1 °C mid-run bends µd(t) at long exposures.

---

## Step 3 — Flat analysis: photon transfer curve → system gain K

**Goal.** Fit the photon transfer curve to extract **K** (system gain).

**Model** (light-induced difference form — the EMVA Eq. 15 evaluation version):
```
σ²y − σ²y.dark = K · (µy − µy.dark)        (Eq. 50, zero-added offset)
```
The absolute variance model (Eq. 15) carries an extra quantization offset σ²q in
the intercept; it cancels exactly in the difference because it's identical in
flat and dark. σ²q reappears in Step 2 (dark floor) and Step 7 (threshold/DR).

**Critical unit warning — the "Var/S = 1/K" trap.** In DN units the shot variance
is *not* K·S but S/K (K in e⁻/DN), so if you fit V = slope·S + b you get
**K = 1/slope**, not the slope itself. State your convention unambiguously. A
unit slip here produced "K = 0.11 e⁻/DN12, full well 460 e⁻" when the truth was
8.45 and 32 ke⁻ (the knee anchor caught it: Nsat/DN_clip = 7.82).

**Correction from real data — fit REGION is everything:**

| Fit range | K12 (e⁻/DN12) |
|---|---|
| Full range (all points, as one naive fit) | 8.46 |
| EMVA 0–70 % of saturation | 8.12 |
| **Clean plateau (mid-signal)** | **7.77** |
| **Knee anchor (Nsat/DN_clip)** | **7.82** |

The physically meaningful K is the **plateau + knee-anchor** value (~7.8), while
a full-range slope is biased high by the soft knee (variance rolls off before
clipping on real CMOS). Basler's 8.4 e⁻/DN report is their EMVA-mandated
*0–70%-of-saturation* regression over the whole range — same knee contamination,
not a different sensor.

**Procedure:**

1. Light-induced values: subtract the dark row at the *same exposure*
   (σ²y − σ²y.dark; µy − µy.dark). At short exposures this is bias
   subtraction; at long ones it also removes dark current.
2. Fit range: from the lowest point to **70 % of saturation** (EMVA rule) —
   *and* drop the soft-knee tail (see the fit-region table above). The bottom
   few points are excluded when read noise is substantial.
3. **Fit a line through the origin on the light-induced difference.** A
   free-offset fit's intercept is *NOT* a usable σ²d cross-check — it mixes
   K²σ²d + σ²q − K·µy.dark and any curvature leaks into it (we measured an
   implied 1974 vs true 156 DN² → garbage). Verify K different ways instead:
4. Cross-checks:
   - **Per-point gain flatness**: K12 = 16·S/dV constant across the
     mid-range (ours 7.72–7.84). A drift = not shot-noise-dominated.
   - **Knee anchor**: K ≈ Nsat/DN_clip (independent, unit-normalized
     calibration) — agrees to ~1 %.
   - **Two-frame vs N-frame**: slopes must match to < ~1 %.
5. **K is wavelength-independent** (shared electronics). One global K serves a
   sensor/hyperspectral camera; η(λ) (Step 5) is the only wavelength dependence.

---

## Step 4 — Saturation capacity (the variance knee)

**Goal.** Locate the saturation point and convert to electrons: **µe.sat**.

**Physics.** At high signal, pixels begin clipping → their variance is censored →
temporal variance **peaks then rolls off** (the knee). µy.sat = mean at the
variance maximum. (EMVA §6.6; use the R4 criterion of ~0.2 % pixels at max, not
a blind argmax.)

**µe.sat = (µy.sat − µy.dark)·K12** (in our e⁻/DN convention) — an *output*
capacity, normally ≤ physical full well since the ADC clips first. (IMX174: the
clip at 4094 DN12 × 7.8 e⁻/DN ≈ 32 ke⁻ = Sony's full well — matched by design.)

**Confirm the hard clip** with 1–2 fully saturated exposures (tvar ≈ 0, mean
pinned at clip: 65,504 DN16 = 4094 DN12; 4095 reserved as flag).

**Cross-check.** Knee anchor K = Nsat/DN_clip — the single most important sanity
check in the pipeline.

**Caveats.** Sampling near the knee matters (sparse points = ±few ke⁻);
release-mandated criterion not blind argmax; a *soft knee* (variance falling
before clipping) is normal CMOS — don't confuse compression with the clip.

---

## Step 5 — Quantum efficiency η (optional: needs calibrated radiometry)

**Goal.** η(λ) — the fraction of incident photons that become electrons.

**Why radiometry.** η = µe/µp; the photon count needs a calibrated light:
```
Φp = E·A_pixel·λ/(h·c)          photons/pixel/s,  µp = Φp·t
```
(Use the **centroid wavelength λc**, not the peak — R4 updated this.)

**Extraction.** µy − µy.dark = R·µp, slope R = K·η; η = R/K (K from Step 3) over
the same 0–70 % range. (Pointwise: η = (µy − µy.dark)/(K·µp).

**Practice (EMVA §9):** monochromatic source (FWHM ≤ 10 nm; up to 50 if reported)
swept across the range; place the *photodiode where the sensor sits*; source
stability is critical (interleave reference reads — drift becomes η error). The
calibration dominates the error budget (Basler ±3.5 % → η absolute ±5–10 %), but
the *spectral shape* is much more accurate (calibration ratios cancel).

**Why it matters:** η(λ) is the only wavelength-dependent noise-model parameter —
dims signal at the same noise floor = fewer photons → electrons. We haven't
measured η on our cameras; vendor curves are the stand-in (Sony peak η 77 %,
Basler 70 % @ 545 nm).

---

## Step 6 — Spatial nonuniformities: DSNU and PRNU

The model adds two fixed spatial patterns (R4 insists "nonuniformity", not
"fixed pattern noise"):
```
s²y = K²·DSNU² + PRNU²·(µy − µy.dark)²     (variances in DN²)
```
DSNU dominates at low signal; PRNU grows quadratically and eventually exceeds
shot noise.

**Extraction.**

1. **DSNU** — spatial std of the (highpass) *dark* image, to e⁻:
   DSNU1288 = sy,dark/K. Free the estimate from temporal contamination:
   L-average (L ≥ 16, better 100–400) with −σ²y/L subtraction, or the
   two-frame covariance (exact, no bookkeeping). IMX174 ROI ≈ 0.6–0.7 e⁻;
   Basler (whole sensor) 1.9 e⁻.
2. **PRNU** — flat at ~50 % saturation:
   ```
   PRNU1288 = √(s²y − s²y.dark) / (µy − µy.dark)
   ```
   IMX174 (highpass) ≈ 0.32–0.37 %; Basler typ 0.5 %. The raw estimate as
   measured is an *upper bound* (diffuser texture is multiplicative, like PRNU).
3. **R4 refinement.** Highpass-filter images first (§8.1 — gradual illumination
   gradients would masquerade as PRNU), and decompose into row/column/pixel
   variances (column noise shows up there; the summary sheet reports all).

**Why averaging (or covariance) is mandatory:** temporal noise (σy ≈ 1 %) is
typically larger than the spatial variation (s ~ 0.3–0.5 %); a single frame's
spatial variance is temporal-noise-dominated. The covariance kills it exactly;
averaging suppresses by √L.

---

## Step 7 — Derived quantities: SNR, dynamic range, threshold

**Goal.** Application-level figures from Steps 2–6: SNR curve, dynamic range,
absolute sensitivity threshold, max SNR.

**Model** (EMVA §2.6, in photons):
```
SNR(µp) = η·µp / √(σ²d + σ²q/K² + η·µp)        (Eq. 21)
```
- Low signal → read-dominated → SNR ∝ µp (slope 1 on log–log)
- High signal → shot-dominated → SNR ∝ √µp (slope ½)

**Validate first.** Overlay the measured points SNR = (µy−µy.dark)/σy on the
model — the global consistency check of every measured parameter. Off-curve point
= a bad parameter or flicker. Your synthetic-noise model must reproduce this.

**Max SNR at saturation.** SNRmax = √Nsat = √(µe.sat). IMX174: √32k = 179 →
45.1 dB (Basler: 45.1 dB — identical).

**Absolute sensitivity threshold** (SNR = 1 — *not* just σd; exact inversion of
Eq. 21, R4 update):
```
µe.min = √(σ²d + σ²q/K² + ¼) + ½ ≥ 1           (Eq. 27)
```
IMX174: σd 6.6 e⁻ → µe.min ≈ 7.0 e⁻; µp.min = µe.min/η ≈ 10 photons @ 545 nm
(Basler: 10).

**Dynamic range:**
```
DR = µsat/µmin = 32k/7.1 ≈ 4,500 → 12.2 bit (Basler: 12.2 bit)
```

**Total SNR** (R4 addition — the single-image, spatial-structure version):
```
SNR_total(µp) = η·µp / √(σd² + DSNU² + PRNU²·(η·µp)² + σq²/K²)    (Eq. 69)
```
The temporal SNR saturates at √Nsat; the **total SNR peaks earlier and then
declines** (PRNU·S grows faster than √S). This is the honest "what you'll see in
one image" curve — the one your synthetic-noise dataset should match for
validation.

**Closing the loop.** Parameters → noise model:
```
DN = round((S_e + shot + dark + read)·(1+PRNU))/K + bias,  clipped at Nsat
```
and Step 7's curves are the acceptance test for anything you generate.

---

## Units & scale conventions (critical)

- The pipeline stores **12-bit data as DN12 << 4 (DN16)**: signals ×16,
  variances ×256, quantization step = 16 DN16.
- Report electron quantities with **K12 (e⁻/DN12)**. For a 12-bit camera:
  K_EMVA (DN/e⁻) = 1/K12, and the EMVA gain formulas in DN/e⁻ use that.
- **EMVA constants with a naked number are in native DN code units** (DN12):
  0.24 / 0.49 / 0.40 / 1/12 are all evaluated in the camera's native ADC unit.
  Example: σ²y.dark < 0.24 DN², σq² = 1/12 DN² — if you feed DN16 storage you'd
  never trigger the quantization rule for any real sensor.
- **Eq. 53/54 read-noise bound when quant-dominated** (σ²d < 0.24 DN):
  ```
  σd < 0.40/K_EMVA = 0.40 × K12     (multiply, not divide — the classic error!)
  ```
  e.g. SPECIM: K12 = 90 → σd < 36 e⁻; the naive 0.40/90 = 0.004 e⁻ violates
  physics (sub-electron noise doesn't exist). σd = 0.40 is Eq. 53 evaluated at
  the worst allowed σ²y.dark with σ/12.

---

## Quantization regimes — when the dark variance isn't read noise

Quantization variance exists in two limits:

- **Regime A (dithering, σ ≳ 1 DN):** quantization error ≈ uniform → variance
  Δ²/12, independent of where the mean sits. This is what EMVA assumes (its
  validity condition: read noise ≥ 1 DN), and what holds for bright flats.
- **Regime B (no dithering, σ ≪ Δ):** quantization variance = **Δ²·f(1−f)**,
  where f = frac(mean/Δ) is the bin position — it oscillates between 0 (mean on
  a boundary, true noise visible) and Δ²/4 (mean mid-bin, maximum masking).
  Result: the measured dark variance can **decrease with exposure** as the mean
  drifts through a bin, producing a NEGATIVE variance-slope "dark current"
  (the SPECIM case, −40..−58 DN²/s), even though the mean-()m-slope dark
  current is real and clean.

Detect it: σ²y.dark < 0.24 DN², or a negative/near-zero variance-slope over time.
When present: the reported σr (even the Δ²/12-corrected cousin) is an **upper
bound**; use the mean-slope dark current; model the dark empirically (the
quantized distribution IS what the camera outputs). The Eq 54 bound applies
(see Units above). Detailed notes: `skill references/quantization-regimes.md`
in the camera-noise-characterization skill.

---

## Pitfall checklist (we paid for these)

1. **Var/S = 1/K, not K** — the gain is the inverse of the slope; the knee
   anchor catches it.
2. **numpy `ddof=0` bias** — temporal variance needs ddof=1 (5 % on 20 frames).
3. **Fit range on the PTC** — full-range slopes are biased high by the soft
   knee; report plateau + knee-anchor K.
4. **Free-offset intercept** is not a usable σ²d cross-check — curvature leaks.
5. **Two-frame vs N-frame MUST agree** (< ~1 %) — a systematic gap is a bug or
   a non-stationary source, not noise.
6. **Black level on Basler** — default user set leaves the dark histogram
   censored at 0 (97 % zeros → σr read 0.06 instead of 6.6 e⁻). Set a digital
   black level ≈ 3×σr above zero, and force GainAuto/ExposureAuto off.
7. **Stray / flicker** — qualify the flat source (frame-mean < 0.1 %, N-vs
   2-frame agreement); PWM is a trap.
8. **Watch the temperature** — report dark current with its temperature; warm up
   to <0.3 °C/min drift.
9. **DN12 vs DN16** — naked EMVA constants are native ADC units; convert your
   storage scale before comparing.
10. **Quantization floor** hides the true read noise on high-gain/large-pixel
    sensors — mark σr as an upper bound and use the empirical dark distribution
    for synthesis.
11. **Firmware auto-black-level while warming** (LP126CU) — heat-soak before
    dark sweeps; a dark mean below the bias reference is an offset step,
    discard it. Watch the per-frame "±X across frames" acquisition line.

---

## camchar CLI quick reference

```
uv sync                      # install (uv-managed, CPython 3.11+)
uv run camchar get-dark-frames  --vendor basler --exposures 1,10,100 ...  # ms!
uv run camchar get-flat-frames  --vendor basler --exposures 1,2,5,...     # ms!
uv run camchar warmup-sensor --vendor thorlabs                            # auto: dark-signal soak
uv run camchar analyze --data data                                        # offline analysis
uv run camchar source-stability-check ...                                 # qualify source BEFORE full run
uv run pytest   # unit tests (offline, mocked backends)
```

- `--exposures` is in **milliseconds** (ms), not seconds: `0.001` = 1 µs and
  clamps to the camera minimum.
- Data layout: `<root>/<vendor>_<model>_(<sensor>)/dark|flat/metadata.json` +
  stack files; `analyze` auto-discovers.
- The two-frame cross-check, ddof=1, dark-subtraction via interpolation, the
  EMVA 0–70 % cap, and the quantization-corrected σr are all in `analyze.py`;
  check the stdout flags and the saved plots under `outputs/<camera>/`.
- For a hyperspectral (SPECIM IQ) run, `uv run camchar analyze --data
  data/SPECIM_IQ` runs the per-band analysis (band_parameters.csv + plots).

## Sources

- EMVA 1288 Release 4.0 Linear (this repo, `docs/`)
- Janesick, *Photon Transfer* (SPIE) — user's copy in ../Library/Physics/Optics
- Skauli, IEEE P4001 hyperspectral-camera characterization draft — framework
  for HSI NESR / A* / resampling noise
- Basler acA1920-155um official EMVA report (link in repo notes)

---

*This guide was distilled from the working session "Specim IQ Noise
Characterization" (Aug 2026). If a step feels wrong in your data — the machinery
is meant to catch that, not to be right.*