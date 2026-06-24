# 🌙 NTL Power Outage Detection
**Detecting large-scale power outages from space using NASA Black Marble nighttime light satellite data.**

Built as part of an IEA × ESA collaboration to support global energy resilience monitoring.

---

## What This Does

When a major disaster strikes — a typhoon, earthquake, or extreme weather event — the lights go out. This pipeline detects those outages automatically by analyzing changes in nighttime light (NTL) radiance from satellite imagery, validated across **58 real-world disaster events** spanning 2017–2025.

---

## Results

| Version | Method | Detection Rate |
|---------|--------|---------------|
| v4 | 2-Pass DOW Baseline + Z-score | **23 / 58 (40%)** |
| v5 | + ERA5 Weather Fusion + Isolation Forest | in progress |

### Detected ✅
Puerto Rico (Hurricane Maria), Fort Myers (Ian), New Orleans (Ida), Turkey (Earthquake), Acapulco (Otis), Myanmar (Mocha), and 17 more.

### Failure Analysis
Undetected cases were systematically categorized — not discarded:

| Failure Type | Count | Root Cause |
|---|---|---|
| Tropical / Monsoon | 16 | Cloud cover & Mie scattering mask the signal |
| No usable data | 10 | Persistent cloud cover, no valid observations |
| Snow / Albedo | 2 | Snow reflection spikes brightness, masking outage |
| Dense canopy | 1 | Tree canopy blocks baseline light signal |
| Other | 6 | Wildfire, conflict, underdeveloped regions |

→ v5 targets the tropical/monsoon gap with ERA5 meteorological fusion.

---

## Methodology

### Data Sources
- **NASA Black Marble VNP46A2** — Daily VIIRS DNB nighttime light, atmosphere-corrected
- **ESA WorldCover v200** — Land cover masking (class 50: built-up infrastructure only)
- **ECMWF ERA5** — Cloud cover, precipitation, snow depth (v5 only)

### Pipeline (v4)

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
Anomaly Detection: Z-score + Drop% thresholds
Singleton detection: Pattern A (drop>80%) / Pattern B (drop>60%, z<-5)
     ↓
Detection Result + Recovery Curve
```

**Why 2-Pass?** A naive baseline computed over the full time series gets contaminated by the outage period itself — pulling the "normal" level down and masking the anomaly. Pass 1 identifies candidate dip periods; Pass 2 recomputes baseline with those periods excluded.

### v5 Additions (ERA5 + ML)
- Extracts daily `total_cloud_cover`, `total_precipitation`, `snow_depth`, `temperature_2m` via GEE
- Builds cloud-corrected radiance drop features
- Isolation Forest for unsupervised anomaly detection on fused features
- Targets tropical/monsoon failure cases (16 of 35 undetected events)

---

## Repository Structure

```
ntl-power-outage-detection/
│
├── src/
│   ├── po_detect_v4.py            # Main detection pipeline
│   ├── po_detect_v4_graph.py      # Visualization module
│   ├── po_detect_v4_non_cy.py     # Non-cyclone event extension
│   └── po_detect_v5_era5.py       # ERA5 + Isolation Forest (v5)
│
├── tools/
│   ├── baseline_comparison.py     # Benchmarks 4 baseline strategies
│   ├── regional_analysis.py       # Per-region threshold analysis & clustering
│   ├── diagnose.py                # False positive root cause diagnosis
│   ├── dow_radiance_analysis.py   # Day-of-week radiance pattern analysis
│   └── verification_workbook.py   # Ground truth matching & news URL generator
│
├── data/
│   └── target_events.csv          # 58 validated disaster events
│
└── requirements.txt
```

---

## Key Design Decisions

**Why Day-of-Week median, not simple rolling average?**
Urban nighttime light has consistent weekly cycles (weekday vs. weekend activity). A simple 30-day rolling average conflates these patterns. DOW median per weekday captures this structure and produces a more stable baseline.

**Why obs_ratio gating?**
On cloudy days, few pixels pass QA — the mean radiance of those few pixels is not representative. Requiring `obs_ratio ≥ 0.4` (≥40% of infrastructure pixels observable) ensures the radiance signal is spatially meaningful before triggering detection.

**Why two separate singleton detection patterns?**
Pattern A (drop>80%, obs>0.5, z<-3.5) catches fast-recovery outages like small island events. Pattern B (drop>60%, obs>0.7, z<-5.0) catches high-confidence extreme drops with more observations. A single threshold would miss one or the other.

---

## Setup

```bash
git clone https://github.com/[your-id]/ntl-power-outage-detection.git
cd ntl-power-outage-detection
pip install -r requirements.txt
```

**Requirements:** Google Earth Engine account with authenticated access (`ee.Authenticate()`).

```python
# Run detection on a single target
python src/po_detect_v4.py
```

Targets are configured in the `targets` list at the bottom of the script.

---

## Stack

`Python` `Google Earth Engine` `NASA Black Marble (VNP46A2)` `ESA WorldCover` `ERA5` `scikit-learn` `pandas` `matplotlib`

---

## Context

Developed during a data science internship at the **International Energy Agency (IEA)**, Paris, as part of a global power outage ground truth dataset project. The pipeline was independently designed, implemented, and validated end-to-end.
