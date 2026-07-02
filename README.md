# 🌙 NTL Power Outage Detection
**Detecting large-scale power outages from space using NASA Black Marble nighttime light satellite data.**

Built as part of an **International Energy Agency (IEA)** × **European Space Agency (ESA)** collaboration to support global energy resilience monitoring.

---

## What This Does

When a major disaster strikes — a typhoon, earthquake, or extreme weather event — the lights go out. This project detects those outages automatically by analyzing changes in nighttime light (NTL) radiance from VIIRS satellite imagery, validated across **58 real-world disaster events** spanning 2017–2025.

---

## Research Journey

This project went through two distinct detection paradigms, each motivated by the failures of the previous one.

### Phase 1: Pixel-level Spatial Detection (first attempt)

The initial approach asked: **"what fraction of infrastructure pixels went dark?"**

Multiple baseline strategies were explored and compared (see `src/pixel_ratio/experiments/`):
- **Sliding**: past 30-day rolling window
- **Sandwich**: pre-outage (−15d) + post-recovery (+15d), avoiding contamination
- **Hybrid**: 50/50 split around the outage window

The sliding approach proved most stable and was refined into a pixel-level DOW (Day-of-Week) baseline with ESA WorldCover infrastructure masking and Robust Z-score anomaly detection.

### Phase 2: Time-series Mean Radiance Detection (main pipeline)

The pixel approach was complex to scale. A simpler, faster alternative was developed: track **mean radiance over infrastructure pixels** against a DOW rolling median baseline.

This became `blackout_tracker.py` — a 2-Pass balanced detector validated across 58 global events, achieving **23/58 (40%) detection rate**.

### The Houston Problem — and why pixel-level still matters

Houston Winter Storm Uri (Feb 2021) was a critical failure case. The time-series approach completely missed it.

**Why?** Snow and ice reflected ambient light, causing post-outage mean radiance to *increase* — the exact opposite of what the detector expected.

The pixel-level approach caught it: even when total brightness goes up due to albedo, pixel-level counting still sees the dark patches across the city.

This motivated the final spatial detector (`blackout_tracker_spatial.py`) combining DOW pixel baseline + ESA WorldCover masking + Robust Z-score.

```
Pixel experiments (sliding/sandwich/hybrid)
          ↓
    Too complex to scale → switched to time-series mean radiance
          ↓
    Achieved 23/58 on global events
          ↓
    Houston Uri: albedo caused mean radiance to INCREASE after outage → missed
          ↓
    Returned to pixel-level with DOW + ESA + Robust Z-score
          ↓
    Houston Uri spike detected ✅  |  Japan Jebi spike detected ✅
```

---

## Results

| Approach | Method | Detection |
|---|---|---|
| Time-series (main) | 2-Pass DOW Baseline + Z-score | **23 / 58 (40%)** |
| Spatial pixel-ratio | DOW pixel baseline + Robust Z-score | Houston ✅, Japan Jebi ✅ |
| ERA5 weather fusion | ERA5 + ML features (in progress) | targeting tropical/monsoon gap |

### Detected (time-series) ✅
Puerto Rico (Maria), Fort Myers (Ian), New Orleans (Ida), Turkey (Earthquake), Acapulco (Otis), Myanmar (Mocha), and 17 more.

### Failure Analysis
Undetected cases were systematically categorized — not discarded:

| Failure Type | Count | Root Cause |
|---|---|---|
| Tropical / Monsoon | 16 | Cloud cover & Mie scattering mask the signal |
| No usable data | 10 | Persistent cloud cover, no valid observations |
| Snow / Albedo | 2 | Snow reflection spikes brightness, masking outage |
| Dense canopy | 1 | Tree canopy blocks baseline light signal |
| Other | 6 | Wildfire, conflict, underdeveloped regions |

→ v5 (ERA5 fusion) targets the tropical/monsoon gap (16 of 35 undetected events).

---

## Methodology

### Data Sources
- **NASA Black Marble VNP46A2** — Daily VIIRS DNB nighttime light, atmosphere-corrected
- **ESA WorldCover v200** — Land cover masking (class 50: built-up infrastructure only)
- **ECMWF ERA5** — Cloud cover, precipitation, snow depth (v5 / in progress)

### Time-series Pipeline (`blackout_tracker.py`)

```
Raw VIIRS NTL
     ↓
QA Filtering (QA ≤ 1, cloud mask, snow flag < 20%)
     ↓
Infrastructure Masking (ESA WorldCover class 50)
     ↓
Pass 1: Day-of-Week Rolling Median Baseline (26-week)
     ↓
Dip Candidate Detection → Mask contaminated periods
     ↓
Pass 2: Clean Baseline (recomputed without outage periods)
     ↓
Anomaly Detection: Robust Z-score + Drop% thresholds
Singleton detection: Pattern A (drop>80%) / Pattern B (drop>60%, z<-5)
     ↓
Detection Result + Recovery Curve
```

**Why 2-Pass?** A naive baseline computed over the full time series gets contaminated by the outage period itself — pulling the "normal" level down and masking the anomaly. Pass 1 identifies candidate dip periods; Pass 2 recomputes baseline with those periods excluded.

**Why Day-of-Week median?** Urban NTL has consistent weekly cycles (weekday vs. weekend). A simple rolling average conflates these. DOW median per weekday produces a more stable, representative baseline.

**Why obs_ratio gating?** On cloudy days, few pixels pass QA — their mean radiance is not spatially representative. Requiring `obs_ratio ≥ 0.4` ensures the signal is meaningful before detection triggers.

### Spatial Pixel-ratio Pipeline (`blackout_tracker_spatial.py`)

```
Raw VIIRS NTL
     ↓
Strict QA Filtering (QA = 0, clear cloud, no snow)
     ↓
ESA WorldCover Class 50 Infrastructure Mask
     ↓
Pixel-level DOW Baseline (median + MAD per weekday, prior year)
     ↓
Per-pixel Robust Z-score: flag pixels below (median - 1.4826 * MAD * k)
     ↓
Spatial Outage Ratio = darkened pixels / total infra pixels (%)
     ↓
Annual trend + event window detection
```

**Why pixel-level?** Mean radiance is distorted by albedo events (snow, ice). Pixel counting is not — even if total brightness increases, dark patches are still visible at the pixel level.

**Why Robust Z-score (MAD)?** Standard deviation is sensitive to outliers. MAD (Median Absolute Deviation) is not, making the threshold more stable across diverse urban environments.

### Recovery Analysis (`blackout_tracker_recovery.py`)

Post-event radiance is smoothed using a Savitzky-Golay filter, then the first date where smoothed radiance exceeds 90% of the pre-event baseline for 7 consecutive days is reported as the recovery date.

---

## Repository Structure

```
ntl-power-outage-detection/
│
├── src/
│   ├── timeseries/                          # Phase 2: Time-series mean radiance
│   │   ├── blackout_tracker.py              # Main detection pipeline (2-Pass DOW)
│   │   ├── blackout_tracker_recovery.py     # + Recovery date detection (SG filter)
│   │   ├── blackout_tracker_noncyclone.py   # Non-cyclone event extension
│   │   └── blackout_tracker_era5.py         # ERA5 weather fusion (in progress)
│   │
│   └── pixel_ratio/                         # Phase 1 & return: Pixel-level detection
│       ├── blackout_tracker_spatial.py      # DOW pixel baseline + ESA + Robust Z-score
│       └── experiments/
│           └── baseline_strategy_test.py    # sliding / sandwich / hybrid comparison
│
├── tools/
│   ├── baseline_comparison.py      # Benchmarks 4 baseline strategies (Simple/DOW/EWMA/STL)
│   ├── regional_analysis.py        # Per-region threshold analysis & clustering
│   ├── diagnose.py                 # False positive root cause diagnosis
│   ├── dow_radiance_analysis.py    # Day-of-week radiance pattern analysis
│   └── verification_workbook.py    # Ground truth matching & news URL generator
│
├── data/
│   └── target_events.csv           # 58 validated disaster events
│
├── config.py                       # GEE initialization (add your project ID)
└── requirements.txt
```

---

## Key Validation Cases

**Houston Winter Storm Uri (Feb 2021) — Albedo failure**

| | Time-series | Spatial pixel-ratio |
|---|---|---|
| Result | ❌ Missed | ✅ Spike detected |
| Reason | Snow reflection inflated mean radiance | Pixel-level counting unaffected by albedo |

**Japan Osaka Typhoon Jebi (Sep 2018)**

| | Time-series | Spatial pixel-ratio |
|---|---|---|
| Result | ❌ Missed | ✅ Spike detected |
| Note | Radiance drop below threshold | FP present — threshold tuning in progress |

---

## Setup

```bash
git clone https://github.com/[your-id]/ntl-power-outage-detection.git
cd ntl-power-outage-detection
pip install -r requirements.txt
```

**Requirements:** Google Earth Engine account with authenticated access.

```bash
# Authenticate GEE (first time only)
python -c "import ee; ee.Authenticate()"

# Edit config.py — add your GEE project ID
# Then run:
python src/timeseries/blackout_tracker.py
```

---

## Stack

`Python` `Google Earth Engine` `NASA Black Marble (VNP46A2)` `ESA WorldCover` `ERA5` `pandas` `matplotlib` `statsmodels` `scipy`
