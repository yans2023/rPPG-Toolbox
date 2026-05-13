# rPPG 1 Minute Indicator Design

## Purpose

This document defines the current 1 minute indicator design for the health model
when the input source is the existing rPPG realtime output in this repository.

The goal is to:

1. Clarify what can be reliably computed from the current rPPG output.
2. Separate core indicators from experimental indicators.
3. Explicitly mark indicators that are not available with the current data.
4. Provide a practical roadmap for future expansion from image/video and BVP.


## Current Input Boundary

The current realtime rPPG scripts save only second-level heart-rate output:

- `elapsed_sec`
- `timestamp`
- `hr_bpm`

Relevant implementation:

- [realtime_hr.py](../../realtime_hr.py)
- [realtime_hr_tscan.py](../../realtime_hr_tscan.py)

In the current saved CSV, the system does **not** persist:

- respiration rate
- beat-level RR/IBI intervals
- BVP waveform
- ROI RGB summary signals
- signal quality score
- motion score

Therefore, the current 1 minute health analysis for rPPG must be limited to
indicators derived from the 1 Hz heart-rate sequence.


## Design Principles

### 1. Input realism first

The current design must only use data that the repository already exports in a
stable way for rPPG.

### 2. Short-window interpretation

1 minute indicators are treated as short-window state indicators or trend
indicators. They should not be presented as equivalent to standard 5 minute HRV
or clinical autonomic assessment.

### 3. Quality-gated output

Every 1 minute window must first pass a minimum data quality requirement before
health interpretation is exposed.

### 4. Explicit separation of certainty

Indicators are split into:

- core indicators
- experimental indicators
- unavailable indicators


## Important Limitation

The existing `healthModel/scripts/complete.py` computes HRV-like metrics from:

- `RR = 60 / HR`

instead of real beat-level RR intervals.

This means the following current 1 minute HRV-related indicators are only
**heart-rate-derived proxy HRV**, not standard ECG-grade or beat-level HRV:

- `MeanJJ`
- `SDNN`
- `RMSSD_HR`
- `HRV_diff`
- `PNN50`
- `SD1`
- `SD2`
- `VAI`
- `VLI`

This limitation must be stated in any downstream report or product output.


## Recommended 1 Minute rPPG Indicator Set

### A. Window metadata

These fields describe the source and validity of one 1 minute window.

- `window_start`
- `window_end`
- `window_size_sec`
- `source`
- `sampling_interval_sec`

Recommended values:

- `window_size_sec = 60`
- `source = rppg_hr_csv`
- `sampling_interval_sec = 1`


### B. Data quality indicators

These indicators should be computed before health interpretation.

- `valid_count`
  - Count of valid heart-rate samples in the 1 minute window.
- `valid_ratio`
  - `valid_count / expected_count`
- `missing_count`
  - Number of missing or invalid samples.
- `hr_outlier_count`
  - Count of samples outside a defined physiological range or showing extreme
    jump artifacts.
- `quality_flag`
  - Recommended values: `good`, `fair`, `poor`

Suggested first-pass quality thresholds:

- `good`: `valid_ratio >= 0.80`
- `fair`: `0.60 <= valid_ratio < 0.80`
- `poor`: `valid_ratio < 0.60`

Recommended interpretation rule:

- If `quality_flag = poor`, do not output health-state conclusions for the
  window.


### C. Core heart-rate level indicators

These are the most stable 1 minute indicators available from current output.

- `avg_heart_rate`
  - Mean heart rate in BPM.
- `median_heart_rate`
  - More robust than mode for a 1 minute window.
- `min_heart_rate`
  - Minimum heart rate in BPM.
- `max_heart_rate`
  - Maximum heart rate in BPM.
- `heart_rate_range`
  - `max_heart_rate - min_heart_rate`

Notes:

- For the 1 minute design, `median_heart_rate` is preferred over the current
  `baseline_heart_rate` mode-based logic.
- `baseline_heart_rate` can still be retained for compatibility, but should not
  be treated as the primary summary statistic.


### D. Core short-window trend indicators

These describe how heart rate is changing within the minute.

- `heart_rate_slope`
  - Linear slope of `hr_bpm` over time, in BPM/minute.
- `heart_rate_delta`
  - Mean of the last segment minus mean of the first segment.
  - Suggested default: last 10 seconds minus first 10 seconds.

These indicators are useful for:

- relaxation vs activation trend
- recovery tendency
- transient instability detection


### E. Core heart-rate stability indicators

These are direct statistics on the heart-rate sequence itself.

- `SD_HR`
  - Standard deviation of `hr_bpm`
- `CV_HR`
  - Coefficient of variation of `hr_bpm`

These are preferred for 1 minute windows because they are more interpretable and
less fragile than short-window spectral metrics.


### F. Core proxy HRV indicators

These come from the current repository logic in
[complete.py](../scripts/complete.py), but must be labeled as proxy HRV.

- `MeanJJ`
  - Mean of `60 / HR`, converted to ms.
- `SDNN`
  - Standard deviation of proxy RR intervals, ms.
- `RMSSD_HR`
  - RMSSD on proxy RR intervals, ms.
- `HRV_diff`
  - Maximum adjacent proxy RR change, ms.

Interpretation guidance:

- Use only as relative short-window variability indicators.
- Do not equate them with standard 5 minute ECG-derived HRV.


### G. Core Poincare indicators

These are also derived from proxy RR intervals.

- `SD1`
- `SD2`
- `SD1_SD2_ratio`

Recommended usage:

- `SD1` is useful as a short-term variation proxy.
- `SD2` may be retained but interpreted cautiously in 1 minute windows.
- `SD1_SD2_ratio` is recommended as a new derived indicator for the 1 minute
  design.


## Experimental 1 Minute Indicators

The following indicators can be computed, but they should be marked as
experimental and should not be used as the sole basis for health interpretation.

- `PNN50`
  - High sensitivity to noise under 1 minute proxy RR.
- `SampEn_HR`
  - Sample entropy on the 1 Hz HR sequence.
  - Sample count is limited in a 60 second window.
- `VAI`
- `VLI`
- `ANS_activity`
  - In the current code this is effectively equivalent to `SDNN`.

Recommended policy:

- Compute them if needed for research.
- Do not expose them as stable user-facing health indicators in the first
  version.


## Indicators Not Available for Current rPPG Output

The following indicators must not be computed from the current rPPG CSV-only
output because the required input does not exist.

### Respiration indicators

Unavailable because current rPPG output does not contain respiration rate or a
respiration waveform.

- `avg_breath_rate`
- `baseline_breath_rate`
- `min_breath_rate`
- `max_breath_rate`
- `RRV_diff`
- `RRV`
- `SDRRI`
- `RMSSD_R`
- `Ve`
- `VLF_R`
- `LF_R`
- `HF_R`
- `LF_HF_R`
- `CV_R`
- `SampEn_R`


### Cardiorespiratory coupling

Unavailable because both heart and respiration sequences are required.

- `RRCC`
- respiration-derived adaptability indicators


### Short-window autonomic frequency interpretation

Not recommended for current 1 minute rPPG CSV-only output.

- `VLF_HR`
- `LF_HR`
- `HF_HR`
- `TP_HR`
- `LF_HF_HR`
- `LF_norm_HR`
- `HF_norm_HR`
- `ANS_balance`
- `vagal_tone`
- `vagal_modulation`
- `sympathetic_tone`
- `sympathetic_modulation`

Reason:

- 1 minute windows are too short for reliable interpretation of these
  frequency-domain autonomic metrics in the current setup.
- The current implementation uses `60 / HR` proxy intervals rather than
  beat-level RR intervals.


### CHRI score

Not recommended for 1 minute rPPG CSV-only output.

Reason:

- Current CHRI depends on `RMSSD_HR`, `SampEn_HR`, and `HF_norm_HR`.
- `HF_norm_HR` is not considered reliable in the present 1 minute design.


## Recommended 1 Minute Output Schema

The following schema is recommended for the first implementation.

```text
window_start
window_end
window_size_sec
source
valid_count
valid_ratio
missing_count
hr_outlier_count
quality_flag
avg_heart_rate
median_heart_rate
min_heart_rate
max_heart_rate
heart_rate_range
heart_rate_slope
heart_rate_delta
SD_HR
CV_HR
MeanJJ
SDNN
RMSSD_HR
HRV_diff
SD1
SD2
SD1_SD2_ratio
PNN50_experimental
SampEn_HR_experimental
```


## Interpretation Guidance

### Suitable for current version

The following health/state interpretations are reasonable for the first version:

- short-window heart-rate level
- short-window heart-rate stability
- short-window activation vs relaxation trend
- short-window variability proxy
- data quality and confidence

### Not suitable for current version

The following should not be claimed from current rPPG CSV-only output:

- respiration state
- cardiorespiratory coupling
- reliable vagal/sympathetic balance
- 5 minute clinical HRV equivalence
- blood pressure estimation
- medical diagnosis


## Implementation Recommendation

### Phase 1: Current CSV-based 1 minute indicators

Implement a 1 minute sliding or fixed-window analysis over:

- `elapsed_sec`
- `timestamp`
- `hr_bpm`

This phase should only produce the recommended schema above.


### Phase 2: Persist richer realtime intermediate outputs

To support better health analysis, the realtime pipeline should additionally
store:

- BVP waveform
- ROI RGB mean traces
- exact frame timestamps
- face box / ROI coordinates
- motion score
- illumination stability score
- signal quality score such as SNR

This is strongly recommended before attempting more advanced indicators.


### Phase 3: Upgrade from HR-derived proxy HRV to beat-level HRV

Once BVP waveform is saved, the system can attempt:

- pulse peak detection
- beat-to-beat interval extraction
- true IBI/PPI sequence generation
- improved RMSSD / SDNN / pNN50
- pulse morphology analysis

This phase is much more valuable than directly adding speculative health scores.


## Long-Term Extension from Image and Signal

The current repository already contains the technical foundation for extracting
more than just heart rate:

- unsupervised rPPG signal processing methods
- deep learning rPPG models
- BVP-oriented datasets
- multitask directions such as BigSmall for pulse, respiration, and facial
  action

### Recommended long-term expansion priorities

#### Priority 1: respiration estimation

Potential input sources:

- low-frequency ROI color variation
- respiratory modulation of BVP amplitude
- respiratory modulation of heart-rate sequence
- head/face/chest subtle motion

Potential outputs:

- respiration rate
- respiration stability
- cardiorespiratory coupling

This is the most realistic next health-physiology extension after heart rate.


#### Priority 2: signal quality and perfusion-related features

Potential outputs:

- BVP SNR
- pulse amplitude stability
- pulse morphology stability
- perfusion-related proxy features

These are valuable because they improve confidence control and model selection.


#### Priority 3: stress / fatigue / autonomic proxy modeling

Potential inputs:

- short-window HR
- proxy HRV or beat-level HRV
- respiration
- motion
- facial action / expression features

Potential outputs:

- stress proxy score
- fatigue proxy score
- arousal / relaxation tendency

These should be framed as wellness or state indicators, not medical diagnosis.


#### Priority 4: SpO2 research

Possible in principle, but difficult with ordinary RGB cameras because of:

- wavelength limitation
- lighting sensitivity
- skin tone sensitivity
- calibration difficulty

This should be treated as research-only unless supported by dedicated hardware
or very carefully controlled data.


#### Priority 5: blood pressure research

Blood pressure is a long-term research topic and should not be treated as a
near-term output.

Possible directions:

- BVP morphology regression
- multi-region pulse timing methods
- multimodal modeling with face motion and pulse features
- person-specific calibration approaches

Key constraints:

- requires dedicated labeled blood pressure data
- often needs calibration
- difficult cross-person generalization
- high risk of overclaiming medical capability

Recommended product policy:

- do not output SBP/DBP in the short term
- if explored, treat it as research-only
- avoid medical claims without separate validation protocol


## Decision Summary

For the current repository state, the 1 minute rPPG health model should use:

- heart-rate level indicators
- heart-rate stability indicators
- short-window trend indicators
- proxy HRV indicators
- explicit quality gating

It should not use:

- respiration indicators
- cardiorespiratory coupling
- short-window autonomic frequency-domain conclusions
- CHRI
- blood pressure estimation


## Next Recommended Engineering Step

The next engineering step should be:

1. implement the 1 minute CSV-based rPPG indicator calculator
2. add BVP and signal-quality persistence to the realtime pipeline
3. then compare 1 minute indicators between rPPG and the reference sensor

