# -*- coding: utf-8 -*-
"""
NTL Power Outage Detector — v4 (Balanced Outage Detector)
==========================================================
Addresses the precision-recall tradeoff issue in v3 (recall collapse)
by relaxing thresholds to a balanced level.

Key changes from v3:
  Fix 1 — Singleton condition diversified (OR patterns)
      Pattern A: Near-total blackout  (drop>80% & obs>0.5 & z<-3.5)
      Pattern B: Strong statistical signal (drop>60% & obs>0.7 & z<-5.0)
  Fix 2 — Sustained detection relaxed: 3-day window, 2-day minimum (was 5/3)
      Allows fast recovery events and partial cloud days
  Fix 3 — obs_ratio threshold relaxed
      Standard detection: 0.5 → 0.4 (recover low-observation regions)
      Tier 2 backdoor:    0.5 → 0.1 (allow thin cloud pass-through in extremes)

Validated against 58 global disaster events (2017–2025).
"""

import sys
import os
import pandas as pd
import numpy as np
import concurrent.futures

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.seasonal import STL

import ee
import urllib.request

# Use a universally available font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# GEE Initialization
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))

try:
    import config
    config.initialize_gee()
except Exception as e:
    print(f"config.py load failed ({e}), falling back to default ee.Initialize().")
    ee.Initialize()


# =============================================================================
# Output directory
# =============================================================================
OUTPUT_DIR = os.path.join(current_dir, '..', '..', 'outputs', 'v4')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Baseline 1: Day-of-Week Rolling Median
# =============================================================================
def dow_median_baseline_masked(series, mask_out=None, window_weeks=26, min_periods=4):
    """
    Compute a day-of-week rolling median baseline.
    Optionally exclude contaminated periods (e.g. outage windows) via mask_out.
    """
    s = series.copy()
    if mask_out is not None:
        s = s.where(~mask_out.fillna(False))

    df = pd.DataFrame({'value': s})
    df['dow'] = df.index.dayofweek
    baseline = pd.Series(index=df.index, dtype=float)

    for dow in range(7):
        mask = df['dow'] == dow
        dow_series = df.loc[mask, 'value']
        rolled = dow_series.shift(1).rolling(window_weeks, min_periods=min_periods).median()
        baseline.loc[mask] = rolled

    return baseline

def dow_median_baseline(series, window_weeks=26):
    return dow_median_baseline_masked(series, mask_out=None, window_weeks=window_weeks)


# =============================================================================
# Baseline 2: STL decomposition (for comparison)
# =============================================================================
def stl_baseline(series):
    """Trend + seasonal baseline via STL decomposition."""
    series_interp = series.interpolate(method='linear').bfill().ffill()
    stl = STL(series_interp, period=7, robust=True)
    res = stl.fit()
    return res.trend + res.seasonal


# =============================================================================
# Robust Z-score
# =============================================================================
def robust_zscore(series, baseline, trim_quantile=0.1):
    """MAD-based robust Z-score, trimming outliers before computing spread."""
    residuals = series - baseline
    res_clean = residuals.dropna()
    if len(res_clean) < 30:
        return pd.Series(index=series.index, dtype=float)

    q_low, q_high = res_clean.quantile([trim_quantile, 1 - trim_quantile])
    trimmed = res_clean[(res_clean >= q_low) & (res_clean <= q_high)]
    mad_val = np.median(np.abs(trimmed - np.median(trimmed)))
    if mad_val == 0 or np.isnan(mad_val):
        mad_val = 0.01
    std_robust = mad_val * 1.4826
    return residuals / std_robust


# =============================================================================
# Core Logic: Balanced 2-Pass Baseline + Detection (v4)
# =============================================================================
def two_pass_baseline_and_detection_v4(df,
                                       z_thresh=-3.0,
                                       drop_thresh_normal=10.0,
                                       drop_thresh_extreme=50.0,
                                       window_weeks=26,
                                       snow_thresh=0.2,
                                       mask_expand_days=7,
                                       sustained_window=3,
                                       sustained_min_days=2,
                                       tier1_obs_thresh=0.4,
                                       tier2_obs_lower=0.1,
                                       tier2_pixel_min=20,
                                       tier2_drop_min=80.0,
                                       tier2_z_max=-4.0):
    """
    2-Pass baseline to avoid outage-period contamination.

    Pass 1: Build initial baseline (may be contaminated by outage signal)
            → identify candidate dip periods
    Pass 2: Recompute baseline with dip periods masked out (clean baseline)
            → run final detection on clean baseline
    """
    radiance = df['radiance_main']
    radiance_for_base = radiance.where(df['valid_for_baseline'])

    # ── Pass 1: contaminated baseline ────────────────────────────────────────
    baseline_p1 = dow_median_baseline_masked(radiance_for_base, mask_out=None, window_weeks=window_weeks)
    drop_p1 = ((baseline_p1 - radiance) / baseline_p1) * 100
    z_p1 = robust_zscore(radiance, baseline_p1)

    snow_ok = df['Snow_Cover_Mean'].fillna(0) < snow_thresh

    tier1_p1 = df['obs_ratio_main'] >= tier1_obs_thresh
    tier2_p1 = (
        (df['obs_ratio_main'] >= tier2_obs_lower)
        & (df['obs_ratio_main'] < tier1_obs_thresh)
        & (df['NTL_Main_Count'] >= tier2_pixel_min)
        & (drop_p1 > tier2_drop_min)
        & (z_p1 < tier2_z_max)
    )
    valid_detect_p1 = tier1_p1 | tier2_p1

    cond_normal_p1  = (valid_detect_p1 & (z_p1 < z_thresh) & (drop_p1 > drop_thresh_normal) & snow_ok)
    cond_extreme_p1 = (valid_detect_p1 & (drop_p1 > drop_thresh_extreme) & (z_p1 < z_thresh) & snow_ok)
    dip_candidate_p1 = (cond_normal_p1 | cond_extreme_p1).fillna(False)

    sustained_p1 = dip_candidate_p1.rolling(window=sustained_window, min_periods=sustained_min_days).sum() >= sustained_min_days

    # Singleton OR patterns (Fix 1)
    pattern_A_p1 = (drop_p1 > 80.0) & (df['obs_ratio_main'] > 0.5) & (z_p1 < -3.5)
    pattern_B_p1 = (drop_p1 > 60.0) & (df['obs_ratio_main'] > 0.7) & (z_p1 < -5.0)
    cond_singleton_p1 = ((pattern_A_p1 | pattern_B_p1) & snow_ok).fillna(False)

    dip_mask_p1 = (sustained_p1 & dip_candidate_p1) | cond_singleton_p1

    # Expand mask to cover surrounding days
    dip_mask_expanded = (dip_mask_p1.astype(int)
                         .rolling(window=mask_expand_days, center=True, min_periods=1).max().astype(bool))

    # ── Pass 2: clean baseline ────────────────────────────────────────────────
    baseline_p2 = dow_median_baseline_masked(radiance_for_base, mask_out=dip_mask_expanded, window_weeks=window_weeks)
    drop_p2 = ((baseline_p2 - radiance) / baseline_p2) * 100
    z_p2 = robust_zscore(radiance, baseline_p2)

    tier1_p2 = df['obs_ratio_main'] >= tier1_obs_thresh
    tier2_p2 = (
        (df['obs_ratio_main'] >= tier2_obs_lower)
        & (df['obs_ratio_main'] < tier1_obs_thresh)
        & (df['NTL_Main_Count'] >= tier2_pixel_min)
        & (drop_p2 > tier2_drop_min)
        & (z_p2 < tier2_z_max)
    )
    valid_detect_p2 = tier1_p2 | tier2_p2

    cond_normal  = (valid_detect_p2 & (z_p2 < z_thresh) & (drop_p2 > drop_thresh_normal) & snow_ok)
    cond_extreme = (valid_detect_p2 & (drop_p2 > drop_thresh_extreme) & snow_ok)
    is_dip_candidate = (cond_normal | cond_extreme).fillna(False)

    sustained = is_dip_candidate.rolling(window=sustained_window, min_periods=sustained_min_days).sum() >= sustained_min_days

    pattern_A_p2 = (drop_p2 > 80.0) & (df['obs_ratio_main'] > 0.5) & (z_p2 < -3.5)
    pattern_B_p2 = (drop_p2 > 60.0) & (df['obs_ratio_main'] > 0.7) & (z_p2 < -5.0)
    cond_singleton_ok = ((pattern_A_p2 | pattern_B_p2) & snow_ok).fillna(False)

    is_dip_main = (sustained & is_dip_candidate) | cond_singleton_ok

    # Store results back into df
    df['baseline_dow_p1']    = baseline_p1
    df['baseline_dow']       = baseline_p2
    df['drop_pct_main_p1']   = drop_p1
    df['drop_pct_main']      = drop_p2
    df['z_main_p1']          = z_p1
    df['z_main']             = z_p2
    df['is_dip_p1']          = dip_mask_p1
    df['is_dip_p1_expanded'] = dip_mask_expanded
    df['is_dip_candidate']   = is_dip_candidate
    df['is_dip_main']        = is_dip_main
    df['caught_by_singleton'] = cond_singleton_ok & ~sustained
    df['caught_by_tier2']     = tier2_p2

    return df


# =============================================================================
# Visualization
# =============================================================================
def plot_diagnostics(df, output_dir, target_name, period_str, event_date=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    title = f'[{target_name} {period_str}] Outage Detection (v4)'
    if event_date is not None:
        title += f' | Event: {pd.Timestamp(event_date).strftime("%Y-%m-%d")}'
    fig.suptitle(title, fontsize=15, fontweight='bold')

    outage_days   = df[df['is_dip_main']]
    singleton_days = df[df['caught_by_singleton']]

    ax = axes[0]
    ax.plot(df.index, df['radiance_main'],
            color='gray', marker='.', markersize=3, linestyle='-', linewidth=0.8, alpha=0.5, label='Observed (QA≤1)')
    ax.plot(df.index, df['baseline_dow_p1'],
            color='lightcoral', linewidth=1.4, linestyle='--', alpha=0.7, label='Baseline Pass 1 (contaminated)')
    ax.plot(df.index, df['baseline_dow'],
            color='navy', linewidth=2.2, label='Baseline Pass 2 (clean, final)')

    if not outage_days.empty:
        ax.scatter(outage_days.index, outage_days['radiance_main'],
                   color='red', edgecolor='black', s=55, zorder=5, label=f'Detected outage ({len(outage_days)} days)')
    if not singleton_days.empty:
        ax.scatter(singleton_days.index, singleton_days['radiance_main'],
                   facecolors='none', edgecolor='purple', s=120, linewidth=2, zorder=6,
                   label=f'Singleton (extreme, {len(singleton_days)} days)')
    if event_date is not None:
        ax.axvline(pd.Timestamp(event_date), color='crimson', linestyle=':', linewidth=2, alpha=0.7, label='Event date')

    ax.set_ylabel('Mean Radiance (nW/cm²/sr)', fontsize=11)
    ax.set_title('Infrastructure Mean Radiance + Baseline', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    ax = axes[1]
    ax.plot(df.index, df['drop_pct_main'],
            color='brown', linewidth=1, marker='.', markersize=3, label='Drop % (Pass 2)')
    ax.axhline(50,  color='red',    linestyle='--', alpha=0.4, label='Drop = 50%')
    ax.axhline(80,  color='purple', linestyle=':',  alpha=0.4, label='Drop = 80% (Pattern A)')
    ax.set_ylabel('Drop (%)', color='brown', fontsize=11)
    ax.tick_params(axis='y', labelcolor='brown')
    ax.grid(True, linestyle='--', alpha=0.4)

    ax2 = ax.twinx()
    ax2.plot(df.index, df['z_main'], color='purple', linewidth=1, alpha=0.7, label='Z-score (Pass 2)')
    ax2.axhline(-3.5, color='orange', linestyle='--', alpha=0.5, label='Z = -3.5 (Pattern A)')
    ax2.axhline(-5.0, color='red',    linestyle=':',  alpha=0.5, label='Z = -5.0 (Pattern B)')
    ax2.set_ylabel('Z-score', color='purple', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='purple')

    if event_date is not None:
        ax.axvline(pd.Timestamp(event_date), color='crimson', linestyle=':', linewidth=2, alpha=0.7)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)
    ax.set_title('Detection Signal (Drop %, Z-score)', fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f"{target_name}_{period_str}_detection_v4.png")
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()


def plot_event_aligned(df, output_dir, target_name, event_date, window_before=30, window_after=60):
    """Plot radiance ratio aligned to event date, showing outage and recovery curve."""
    event_date = pd.Timestamp(event_date)
    start = event_date - pd.Timedelta(days=window_before)
    end   = event_date + pd.Timedelta(days=window_after)
    sub   = df.loc[start:end].copy()
    if sub.empty:
        return None

    sub['days_from_event'] = (sub.index - event_date).days
    sub['radiance_ratio']  = sub['radiance_main'] / sub['baseline_dow']

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(sub['days_from_event'], sub['radiance_ratio'],
            color='navy', linewidth=1.5, marker='o', markersize=4, label='Radiance ratio (Pass 2)')
    ax.axhline(1.0, color='gray',   linestyle='-',  linewidth=0.8, alpha=0.5)
    ax.axhline(0.9, color='red',    linestyle='--', alpha=0.5, label='90% line')
    ax.axvline(0,   color='crimson', linestyle=':',  linewidth=2, label='Event date')
    ax.fill_between(sub['days_from_event'], 0, sub['radiance_ratio'],
                    where=(sub['radiance_ratio'] < 0.9), alpha=0.25, color='red')

    ax.set_xlabel('Days from Event', fontsize=12)
    ax.set_ylabel('Radiance Ratio (observed / baseline)', fontsize=12)
    ax.set_title(f'[{target_name}] Event-aligned Recovery Curve (v4)', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, max(1.3, sub['radiance_ratio'].max() * 1.1))

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{target_name}_event_aligned_v4.png")
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()

    checkpoints = [-7, -1, 1, 3, 7, 14, 30, 60]
    snapshot = {}
    for d in checkpoints:
        row = sub[sub['days_from_event'] == d]
        snapshot[f'D{d:+d}'] = float(row['radiance_ratio'].iloc[0]) if not row.empty else np.nan
    return snapshot


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_detection(df, event_date, expected_window_days=14):
    """Check whether the pipeline detected an outage within the expected window."""
    event_date = pd.Timestamp(event_date)
    window_end = event_date + pd.Timedelta(days=expected_window_days)
    window = df.loc[event_date:window_end]

    if window.empty:
        return {'detected': False, 'lag_days': None, 'max_drop_pct': None, 'min_radiance_ratio': None}

    detected = bool(window['is_dip_main'].any())
    lag = None
    if detected:
        first_dip = window[window['is_dip_main']].index[0]
        lag = (first_dip - event_date).days

    max_drop  = float(window['drop_pct_main'].max()) if window['drop_pct_main'].notna().any() else None
    min_ratio = float((window['radiance_main'] / window['baseline_dow']).min()) if window['baseline_dow'].notna().any() else None

    return {'detected': detected, 'lag_days': lag, 'max_drop_pct': max_drop, 'min_radiance_ratio': min_ratio}


# =============================================================================
# Satellite Image Download
# =============================================================================
def download_dip_images(col, dip_dates, roi, name, output_dir, max_images=10):
    if not dip_dates:
        return
    if len(dip_dates) > max_images:
        idx = np.linspace(0, len(dip_dates) - 1, max_images, dtype=int)
        dip_dates = [dip_dates[i] for i in idx]

    image_dir = os.path.join(output_dir, f"{name}_dip_images_v4")
    os.makedirs(image_dir, exist_ok=True)
    vis_params = {
        'bands': ['DNB_BRDF_Corrected_NTL'],
        'min': 0, 'max': 60,
        'palette': ['000000', '0000FF', '800080', 'FFFF00', 'FFFFFF']
    }
    region_coords = roi.bounds().getInfo()['coordinates']

    for date_ts in dip_dates:
        date_str     = date_ts.strftime('%Y-%m-%d')
        next_day_str = (date_ts + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        img = col.filterDate(date_str, next_day_str).first()
        try:
            url      = img.visualize(**vis_params).getThumbURL({'region': region_coords, 'dimensions': 512, 'format': 'png'})
            img_path = os.path.join(image_dir, f"{name}_{date_str}.png")
            urllib.request.urlretrieve(url, img_path)
        except Exception:
            pass


# =============================================================================
# Main Analysis Function
# =============================================================================
def analyze_outage(target_info):
    name       = target_info['name']
    lat        = target_info['lat']
    lon        = target_info['lon']
    start_year = target_info['start_year']
    end_year   = target_info['end_year']
    event_date = target_info.get('event_date', None)
    buffer_km  = target_info.get('buffer_km', 30)
    period_str = f"{start_year}_{end_year}"

    roi        = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)
    start_date = f"{start_year - 1}-07-01"
    end_date   = f"{end_year + 1}-03-01"
    col        = ee.ImageCollection("NASA/VIIRS/002/VNP46A2").filterBounds(roi).filterDate(start_date, end_date)

    # Load ESA WorldCover infrastructure mask (class 50 = built-up)
    try:
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").filterBounds(roi).first()
        infra_mask = worldcover.select('Map').eq(50)
    except Exception as e:
        print(f"[{name}] WorldCover load failed: {e}")
        return None

    def extract_features(img):
        qa   = img.select('Mandatory_Quality_Flag')
        snow = img.select('Snow_Flag')
        mask_main   = qa.lte(1)
        mask_strict = qa.eq(0)

        total_infra = ee.Image.constant(1).updateMask(infra_mask).rename('Total_Infra')
        ntl         = img.select('DNB_BRDF_Corrected_NTL').updateMask(infra_mask)
        ntl_main    = ntl.updateMask(mask_main).rename('NTL_Main')
        ntl_strict  = ntl.updateMask(mask_strict).rename('NTL_Strict')
        snow_info   = snow.updateMask(infra_mask).rename('Snow_Cover')

        sum_count   = ee.Reducer.sum().combine(reducer2=ee.Reducer.count(), sharedInputs=True)
        stats_light = ee.Image([total_infra, ntl_main, ntl_strict]).reduceRegion(
            reducer=sum_count, geometry=roi, scale=500, maxPixels=1e9)
        stats_snow  = snow_info.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=500, maxPixels=1e9)

        return ee.Feature(None, {
            'date':              img.date().format('YYYY-MM-dd'),
            'Total_Infra_Count': stats_light.get('Total_Infra_count'),
            'NTL_Main_Sum':      stats_light.get('NTL_Main_sum'),
            'NTL_Main_Count':    stats_light.get('NTL_Main_count'),
            'NTL_Strict_Sum':    stats_light.get('NTL_Strict_sum'),
            'NTL_Strict_Count':  stats_light.get('NTL_Strict_count'),
            'Snow_Cover_Mean':   stats_snow.get('Snow_Cover'),
        })

    print(f"[{name}] Extracting data ({start_date} to {end_date})...")
    try:
        feats = col.map(extract_features).getInfo()['features']
    except Exception as e:
        print(f"[{name}] GEE error: {e}")
        return None

    df_full = pd.DataFrame([f['properties'] for f in feats])
    if df_full.empty:
        return None

    df_full['date'] = pd.to_datetime(df_full['date'])
    df_full = df_full.set_index('date').sort_index().asfreq('D')

    numeric_cols = ['Total_Infra_Count', 'NTL_Main_Sum', 'NTL_Main_Count',
                    'NTL_Strict_Sum', 'NTL_Strict_Count', 'Snow_Cover_Mean']
    df_full[numeric_cols] = df_full[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df_full['obs_ratio_main']   = df_full['NTL_Main_Count']   / df_full['Total_Infra_Count']
    df_full['obs_ratio_strict'] = df_full['NTL_Strict_Count'] / df_full['Total_Infra_Count']
    df_full[['obs_ratio_main', 'obs_ratio_strict']] = (
        df_full[['obs_ratio_main', 'obs_ratio_strict']].replace([np.inf, -np.inf], np.nan))

    df_full['valid_for_baseline'] = df_full['obs_ratio_main'] >= 0.6
    df_full['valid_for_detect']   = df_full['obs_ratio_main'] >= 0.4

    df_full['radiance_main']   = df_full['NTL_Main_Sum']   / df_full['NTL_Main_Count']
    df_full['radiance_strict'] = df_full['NTL_Strict_Sum'] / df_full['NTL_Strict_Count']
    df_full['weekday_name']    = df_full.index.day_name()

    radiance_for_base    = df_full['radiance_main'].where(df_full['valid_for_baseline'])
    df_full['baseline_stl'] = stl_baseline(radiance_for_base)

    df_full = two_pass_baseline_and_detection_v4(
        df_full,
        z_thresh=-3.0, drop_thresh_normal=10.0, drop_thresh_extreme=50.0,
        window_weeks=26, snow_thresh=0.2, mask_expand_days=7,
        sustained_window=3, sustained_min_days=2,
        tier1_obs_thresh=0.4, tier2_obs_lower=0.1
    )

    df = df_full.loc[f"{start_year}":f"{end_year}"].copy()

    n_dip_main  = int(df['is_dip_main'].sum())
    n_singleton = int(df['caught_by_singleton'].sum())
    print(f"[{name}] Detected outage days: {n_dip_main} (singleton: {n_singleton})")

    eval_result = None
    if event_date is not None:
        eval_result = evaluate_detection(df, event_date, expected_window_days=14)

    # Save CSV
    csv_cols = ['Total_Infra_Count', 'NTL_Main_Count', 'obs_ratio_main', 'radiance_main',
                'baseline_dow', 'drop_pct_main', 'z_main', 'Snow_Cover_Mean', 'weekday_name',
                'valid_for_detect', 'is_dip_candidate', 'is_dip_main',
                'caught_by_singleton', 'caught_by_tier2']
    csv_path = os.path.join(OUTPUT_DIR, f"{name}_{period_str}_diagnostic_v4.csv")
    df[csv_cols].to_csv(csv_path)

    plot_diagnostics(df, OUTPUT_DIR, name, period_str, event_date=event_date)

    snapshot = None
    if event_date is not None:
        snapshot = plot_event_aligned(df, OUTPUT_DIR, name, event_date)

    download_dip_images(col, list(df[df['is_dip_main']].index), roi, name, OUTPUT_DIR, max_images=10)

    result = {'name': name, 'event_date': event_date,
              'n_dip_days': n_dip_main, 'n_singleton': n_singleton}
    if eval_result is not None: result.update(eval_result)
    if snapshot   is not None: result.update(snapshot)

    return result


# =============================================================================
# Summary Report
# =============================================================================
def write_spec_sheet(results, output_dir):
    rows = [r for r in results if r is not None]
    if not rows:
        return
    df_spec   = pd.DataFrame(rows)
    spec_path = os.path.join(output_dir, "detection_spec_sheet_v4.csv")
    df_spec.to_csv(spec_path, index=False)
    print(f"\nDetection spec sheet saved: {spec_path}")

    detected = df_spec.get('detected', pd.Series([], dtype=bool))
    if not detected.empty:
        print(f"\n{'='*60}")
        print(f"v4 Summary (Balanced Detector)")
        print(f"{'='*60}")
        print(f"Detected:   {int(detected.sum())} / {len(df_spec)}")
        failed = df_spec[~df_spec['detected'].fillna(False)]
        if not failed.empty:
            print(f"\nMissed cases ({len(failed)}):")
            for _, row in failed.iterrows():
                print(f"  - {row['name']} ({row['event_date']})")
        print(f"{'='*60}")


# =============================================================================
# Entry Point — configure targets here
# =============================================================================
if __name__ == "__main__":
    targets = [
        # Add or remove targets as needed
        # Format: name, lat, lon, start_year, end_year, event_date (YYYY-MM-DD)
        {"name": "PuertoRico_SanJuan_Maria",  "lat": 18.4655, "lon": -66.1057, "start_year": 2017, "end_year": 2017, "event_date": "2017-09-20"},
        {"name": "USA_FL_FortMyers_Ian",      "lat": 26.6406, "lon": -81.8723, "start_year": 2022, "end_year": 2022, "event_date": "2022-09-28"},
        {"name": "Mexico_Acapulco_Otis",      "lat": 16.8531, "lon": -99.8237, "start_year": 2023, "end_year": 2023, "event_date": "2023-10-25"},
        {"name": "USA_LA_NewOrleans_Ida",     "lat": 29.9511, "lon": -90.0715, "start_year": 2021, "end_year": 2021, "event_date": "2021-08-29"},
        {"name": "Turkey_Antakya_Earthquake", "lat": 36.2021, "lon": 36.1603,  "start_year": 2023, "end_year": 2023, "event_date": "2023-02-06"},
    ]

    print("NTL Power Outage Detection — v4 (Balanced)")
    print("=" * 60)
    print(f"Targets: {len(targets)}")
    print("Fix 1: Singleton OR patterns (A: drop>80 / B: drop>60)")
    print("Fix 2: Sustained window 3-day, min 2-day")
    print("Fix 3: obs_ratio relaxed to 0.4 / Tier2 backdoor 0.1")
    print("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(analyze_outage, targets))

    write_spec_sheet(results, OUTPUT_DIR)
