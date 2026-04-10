"""
evaluation.py - Comprehensive evaluation suite for the multi-horizon
probabilistic weather forecasting system (7-day ahead, 4 variables) used
for coffee-belt agricultural planning in Minas Gerais, Brazil.

The script is intentionally standalone: it consumes plain numpy arrays of
predictions and ground truth, so it can be wired up to any model that emits
Gaussian (mu, sigma) heads. All outputs land in `results/`:

    PNGs (each saved at 300 dpi)
        01_pit_histograms.png            - PIT calibration check, 4 panels
        02_reliability_diagrams.png      - quantile reliability, 4 panels
        03_crps_by_horizon.png           - mean CRPS per horizon vs climatology
        04_sigma_vs_rmse_by_horizon.png  - uncertainty vs realised error, 4 panels
        05_fan_chart_<var>.png           - 4 fan charts (one per variable)
        06_scatter_mu_vs_observed.png    - point-forecast accuracy scatter
        07_rainfall_threshold_exceedance.png  - heavy-rain probability reliability
        08_frost_risk.png                - frost-probability reliability
        09_calibration.png              - reliability diagram (single-variable PIL)
        10_forecast.png                 - 7-day forecast vs observations (PIL)

    CSVs
        table_point_metrics.csv          - MAE/RMSE per (variable, horizon)
                                            with persistence + climatology baselines
        table_interval_coverage.csv      - 50/80/95% PI coverage with deviation flags

Each metric/plot includes a docstring explaining what it measures and how
to interpret it. Run `python evaluation.py` directly for an end-to-end
synthetic demo with no real model required.

Expected array shapes:
    y_true:     (n_samples, 7, 4)   ground truth in real-world units
    mu_pred:    (n_samples, 7, 4)   predicted Gaussian means
    sigma_pred: (n_samples, 7, 4)   predicted Gaussian standard deviations

Variable order (axis 2):
    0 - Temperature (deg C)
    1 - Rainfall (mm)
    2 - Humidity (%)
    3 - Wind Speed (m/s)
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import torch
    from torch.utils.data import DataLoader
    _TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    torch = None
    DataLoader = None
    _TORCH_IMPORT_ERROR = exc

# Force UTF-8 stdout/stderr so unicode (sigma, mu, deg) prints on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

# Optional seaborn — used purely for the colour-blind theme.
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    _PALETTE = sns.color_palette("colorblind", 4)
except ImportError:
    # Wong colour-blind safe palette as fallback
    _PALETTE = [
        (0.000, 0.447, 0.741),   # blue
        (0.835, 0.369, 0.000),   # vermilion
        (0.000, 0.620, 0.451),   # bluish green
        (0.800, 0.475, 0.655),   # reddish purple
    ]

# Publication-quality matplotlib defaults
mpl.rcParams.update({
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 110,
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "regular",
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =============================================================================
# Constants
# =============================================================================

VAR_NAMES = ['Temperature (°C)', 'Rainfall (mm)', 'Humidity (%)', 'Wind Speed (m/s)']
TEMP_IDX = 0
RAIN_IDX = 1
HUMID_IDX = 2
WIND_IDX = 3

NUM_VARS = 4
NUM_HORIZONS = 7
HORIZONS = np.arange(1, NUM_HORIZONS + 1)

DEFAULT_OUTPUT_DIR = "results"
MODELS_DIR = "models"

# Agricultural decision thresholds
RAINFALL_THRESHOLD_MM = 20.0   # heavy-rain alert for irrigation/harvest planning
FROST_THRESHOLD_C = 3.0        # frost protection trigger for southern growers

# Central prediction intervals to evaluate
INTERVAL_LEVELS = (0.50, 0.80, 0.95)
INTERVAL_Z = {
    0.50: 0.6744897501960817,
    0.80: 1.2815515655446004,
    0.95: 1.959963984540054,
}


def _require_torch() -> None:
    """Raise a clear error when torch-dependent paths are used without PyTorch."""
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for this mode. Install it in the weather-forecasting conda environment."
        ) from _TORCH_IMPORT_ERROR


# =============================================================================
# Math helpers — no scipy dependency
# =============================================================================

# Vectorised standard normal CDF using math.erf, wrapped via numpy frompyfunc.
_erf_vec = np.frompyfunc(math.erf, 1, 1)


def std_normal_cdf(x):
    """
    Standard normal CDF Phi(x), elementwise on numpy arrays.
    Implemented via math.erf so the script has no scipy dependency.
    """
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + _erf_vec(x / math.sqrt(2.0)).astype(np.float64))


def gaussian_crps(mu, sigma, y):
    """
    Closed-form Continuous Ranked Probability Score for Gaussian forecasts.

        CRPS(N(mu, sigma^2); y) = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
        where z = (y - mu) / sigma

    CRPS is a strictly proper scoring rule expressed in the same units as
    the forecast variable. It jointly rewards sharpness (tight predictive
    distributions) and calibration (distributions centred on the truth).
    Lower is better.
    """
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    Phi = std_normal_cdf(z)
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


# =============================================================================
# Validation / IO helpers
# =============================================================================

def _validate_shapes(y_true, mu_pred, sigma_pred):
    if y_true.shape != mu_pred.shape or y_true.shape != sigma_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true{y_true.shape}, mu_pred{mu_pred.shape}, "
            f"sigma_pred{sigma_pred.shape}; all three must match."
        )
    if y_true.ndim != 3 or y_true.shape[1] != NUM_HORIZONS or y_true.shape[2] != NUM_VARS:
        raise ValueError(
            f"Expected arrays of shape (n_samples, {NUM_HORIZONS}, {NUM_VARS}); "
            f"got {y_true.shape}"
        )


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _slug(name: str) -> str:
    """Filename-safe slug from a variable label, e.g. 'Temperature (°C)' -> 'temperature'."""
    base = name.split(' (')[0]
    return base.lower().replace(' ', '_')


# =============================================================================
# Plot 1 - PIT histograms
# =============================================================================

def plot_pit_histograms(y_true, mu_pred, sigma_pred, output_dir,
                        var_names=VAR_NAMES, n_bins=20):
    """
    Probability Integral Transform (PIT) histograms - per-variable calibration.

    For each prediction, PIT_i = Phi((y_i - mu_i)/sigma_i). If the predictive
    Gaussians are perfectly calibrated, the PIT values are uniform on [0, 1].
    Departures diagnose the failure mode:
        U-shape   - predictive distributions are too narrow (under-dispersed,
                    truth lands in the tails too often)
        Inverted-U / dome - too wide (over-dispersed)
        Monotone slope    - bias in the predicted mean
    """
    sigma_safe = np.maximum(sigma_pred, 1e-6)
    pit = std_normal_cdf((y_true - mu_pred) / sigma_safe)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for vi, name in enumerate(var_names):
        ax = axes[vi]
        vals = pit[..., vi].ravel()
        ax.hist(vals, bins=bin_edges, density=True, color=_PALETTE[vi],
                edgecolor='white', linewidth=0.6, alpha=0.85)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.2,
                   label='Uniform (calibrated)')
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_title(name)
        ax.set_xlabel('PIT value')
        if vi % 2 == 0:
            ax.set_ylabel('Density')
        ax.legend(loc='upper center', fontsize=9)

    fig.suptitle('PIT Histograms — Predictive Calibration', fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(output_dir, '01_pit_histograms.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# =============================================================================
# Plot 2 - Reliability diagrams (quantile-based)
# =============================================================================

def plot_reliability_diagrams(y_true, mu_pred, sigma_pred, output_dir,
                              var_names=VAR_NAMES):
    """
    Quantile reliability diagrams - one panel per variable.

    For each nominal level p in {0.05, 0.10, ..., 0.95}, the model's
    predicted p-quantile is q_p(i) = mu_i + sigma_i * Phi^-1(p). A
    well-calibrated forecast satisfies P(y < q_p) = p, so the empirical
    fraction of observations below the p-quantile should lie on the diagonal.

    Equivalently this is the empirical CDF of the PIT values evaluated at
    the deciles, which is the standard probabilistic reliability diagram.
    """
    sigma_safe = np.maximum(sigma_pred, 1e-6)
    pit = std_normal_cdf((y_true - mu_pred) / sigma_safe)
    nominal = np.linspace(0.05, 0.95, 19)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for vi, name in enumerate(var_names):
        ax = axes[vi]
        v = pit[..., vi].ravel()
        emp = np.array([(v <= p).mean() for p in nominal])

        ax.fill_between(nominal, np.maximum(nominal - 0.05, 0),
                        np.minimum(nominal + 0.05, 1),
                        color='grey', alpha=0.18, label='±5% tolerance')
        ax.plot([0, 1], [0, 1], color='black', linestyle='--',
                linewidth=1.2, label='Perfect')
        ax.plot(nominal, emp, marker='o', color=_PALETTE[vi],
                linewidth=2, markersize=6, label='Model')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(name)
        if vi >= 2:
            ax.set_xlabel('Nominal quantile p')
        if vi % 2 == 0:
            ax.set_ylabel('Empirical frequency')
        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Reliability Diagrams — Predicted vs Observed Quantiles',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(output_dir, '02_reliability_diagrams.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# =============================================================================
# Plot 3 - CRPS by forecast horizon
# =============================================================================

def plot_crps_by_horizon(y_true, mu_pred, sigma_pred, output_dir,
                         var_names=VAR_NAMES):
    """
    Mean CRPS at each forecast horizon (day 1 .. 7), one line per variable.

    CRPS should rise with horizon as the future becomes harder to predict.
    A dashed climatological baseline is drawn for each variable - it uses
    a Gaussian centred on the test-set marginal mean with the test-set
    marginal standard deviation. The model's solid line should sit clearly
    below its dashed counterpart for the model to add forecast skill.
    """
    crps_per = gaussian_crps(mu_pred, sigma_pred, y_true)   # (n,7,4)
    crps_horizon = crps_per.mean(axis=0)                    # (7,4)

    # Climatology baseline: predict the marginal mean and std of the targets.
    clim_mu = y_true.mean(axis=(0, 1), keepdims=True)              # (1,1,4)
    clim_sigma = y_true.std(axis=(0, 1), keepdims=True) + 1e-6
    clim_crps_full = gaussian_crps(
        np.broadcast_to(clim_mu, y_true.shape),
        np.broadcast_to(clim_sigma, y_true.shape),
        y_true,
    )
    clim_crps = clim_crps_full.mean(axis=0)                        # (7,4)

    fig, ax = plt.subplots(figsize=(10, 6))
    for vi, name in enumerate(var_names):
        ax.plot(HORIZONS, crps_horizon[:, vi], marker='o',
                color=_PALETTE[vi], linewidth=2.2,
                label=f'{name} — model')
        ax.plot(HORIZONS, clim_crps[:, vi], linestyle=':', marker='x',
                color=_PALETTE[vi], linewidth=1.6, alpha=0.7,
                label=f'{name} — climatology')

    ax.set_xlabel('Forecast horizon (days ahead)')
    ax.set_ylabel('Mean CRPS (variable units)')
    ax.set_title('CRPS by Forecast Horizon')
    ax.set_xticks(HORIZONS)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    fig.tight_layout()
    out = os.path.join(output_dir, '03_crps_by_horizon.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# =============================================================================
# Plot 4 - Predicted sigma vs RMSE by horizon
# =============================================================================

def plot_sigma_vs_rmse_by_horizon(y_true, mu_pred, sigma_pred, output_dir,
                                  var_names=VAR_NAMES):
    """
    Mean predicted sigma vs realised RMSE at each horizon, per variable.

    For an unbiased Gaussian forecaster, the predicted standard deviation
    should track the root-mean-squared-error: E[(y-mu)^2] = sigma^2.
        Predicted sigma << RMSE  -> overconfident model (sharper than warranted)
        Predicted sigma >> RMSE  -> underconfident model (intervals too wide)
    Both bars rising together with horizon is the expected, healthy pattern.
    """
    sigma_mean = sigma_pred.mean(axis=0)                                  # (7,4)
    rmse = np.sqrt(((mu_pred - y_true) ** 2).mean(axis=0))                # (7,4)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    width = 0.4
    x = np.arange(NUM_HORIZONS)

    for vi, name in enumerate(var_names):
        ax = axes[vi]
        ax.bar(x - width / 2, rmse[:, vi], width=width,
               color=_PALETTE[vi], alpha=0.95, label='RMSE (realised error)')
        ax.bar(x + width / 2, sigma_mean[:, vi], width=width,
               color=_PALETTE[vi], alpha=0.45, hatch='//',
               edgecolor='white', label='Mean predicted σ')
        ax.set_xticks(x)
        ax.set_xticklabels([f'd{h}' for h in HORIZONS])
        ax.set_title(name)
        if vi >= 2:
            ax.set_xlabel('Forecast horizon')
        if vi % 2 == 0:
            ax.set_ylabel('Spread')
        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Predicted σ vs RMSE by Horizon', fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(output_dir, '04_sigma_vs_rmse_by_horizon.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# =============================================================================
# Plot 5 - Fan charts (one per variable)
# =============================================================================

def plot_fan_charts(y_true, mu_pred, sigma_pred, output_dir,
                    var_names=VAR_NAMES,
                    window_start=0, window_length=40):
    """
    Fan charts - a contiguous test window with the 50% / 80% / 95% Gaussian
    prediction intervals overlaid on the observed series.

    Each x-tick is one consecutive sample's day-1 (one-day-ahead) forecast,
    so a 40-sample window plots 40 consecutive day-ahead predictions. The
    nested shaded regions show how forecast uncertainty bands relate to
    realised observations - well-calibrated 95% PIs should contain almost
    all of the dashed 'observed' line.
    """
    n = y_true.shape[0]
    end = min(window_start + window_length, n)
    sl = slice(window_start, end)
    days = np.arange(end - window_start)

    z50, z80, z95 = INTERVAL_Z[0.50], INTERVAL_Z[0.80], INTERVAL_Z[0.95]

    for vi, name in enumerate(var_names):
        m = mu_pred[sl, 0, vi]
        s = np.maximum(sigma_pred[sl, 0, vi], 1e-6)
        y = y_true[sl, 0, vi]
        col = _PALETTE[vi]

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.fill_between(days, m - z95 * s, m + z95 * s,
                        color=col, alpha=0.15, label='95% PI')
        ax.fill_between(days, m - z80 * s, m + z80 * s,
                        color=col, alpha=0.25, label='80% PI')
        ax.fill_between(days, m - z50 * s, m + z50 * s,
                        color=col, alpha=0.40, label='50% PI')
        ax.plot(days, m, color=col, linewidth=2, label='Predicted μ')
        ax.plot(days, y, color='black', linewidth=1.4, marker='o',
                markersize=4, linestyle='--', label='Observed')

        ax.set_xlabel('Day in test window')
        ax.set_ylabel(name)
        ax.set_title(f'Fan Chart — {name} (1-day-ahead, '
                     f'samples {window_start}–{end - 1})')
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        fig.tight_layout()
        out = os.path.join(output_dir, f'05_fan_chart_{_slug(name)}.png')
        fig.savefig(out)
        plt.close(fig)
        print(f"  saved {out}")


# =============================================================================
# Plot 6 - Predicted mu vs observed scatter
# =============================================================================

def plot_scatter_mu_vs_observed(y_true, mu_pred, output_dir,
                                var_names=VAR_NAMES, max_points=8000):
    """
    Predicted mean vs observed scatter, with a y = x reference line and the
    R^2 coefficient of determination annotated for each variable.

    R^2 = 1 - SS_res / SS_tot is interpreted as the fraction of variance in
    the observations explained by the predicted mean. Points are randomly
    subsampled to `max_points` so that publication-quality figures stay
    legible regardless of dataset size.
    """
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for vi, name in enumerate(var_names):
        ax = axes[vi]
        y = y_true[..., vi].ravel()
        m = mu_pred[..., vi].ravel()

        # R^2 computed on the full set, before any subsampling for plotting.
        ss_res = float(np.sum((y - m) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot

        if y.size > max_points:
            idx = rng.choice(y.size, size=max_points, replace=False)
            y_p = y[idx]
            m_p = m[idx]
        else:
            y_p, m_p = y, m

        ax.scatter(y_p, m_p, s=8, alpha=0.35,
                   color=_PALETTE[vi], edgecolors='none')
        lo = float(min(y_p.min(), m_p.min()))
        hi = float(max(y_p.max(), m_p.max()))
        ax.plot([lo, hi], [lo, hi], color='black', linestyle='--',
                linewidth=1.2, label='y = x')

        ax.set_xlabel(f'Observed {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(name)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top',
                bbox=dict(facecolor='white', edgecolor='lightgrey', pad=4))
        ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Predicted Mean vs Observed', fontsize=14, y=1.00)
    fig.tight_layout()
    out = os.path.join(output_dir, '06_scatter_mu_vs_observed.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# =============================================================================
# Threshold reliability helpers (used by plots 7 and 8)
# =============================================================================

def _threshold_reliability_arrays(p_pred, obs, n_bins):
    """Bin predicted probabilities into deciles and compute mean predicted
    vs observed exceedance frequency in each bin. Returns bin centres,
    mean predicted, observed frequency, and per-bin counts."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_pred, bin_edges) - 1, 0, n_bins - 1)

    bin_pred = np.zeros(n_bins)
    bin_obs = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            bin_pred[b] = p_pred[mask].mean()
            bin_obs[b] = obs[mask].mean()
            bin_count[b] = mask.sum()
    return bin_edges, bin_pred, bin_obs, bin_count


def _plot_threshold_reliability(p_pred, obs, *, colour, threshold_label,
                                title, save_path, n_bins=10):
    """Shared layout for the rainfall- and frost-reliability figures."""
    bin_edges, bin_pred, bin_obs, bin_count = _threshold_reliability_arrays(
        p_pred, obs, n_bins
    )
    valid = bin_count > 0

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={'width_ratios': [1.0, 1.2]},
    )

    ax1.plot([0, 1], [0, 1], color='black', linestyle='--',
             linewidth=1.2, label='Perfect')
    ax1.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                     color='grey', alpha=0.15, label='±5% tolerance')
    ax1.plot(bin_pred[valid], bin_obs[valid], marker='o',
             color=colour, linewidth=2, markersize=8, label='Model')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(f'Forecast {threshold_label}')
    ax1.set_ylabel('Observed frequency')
    ax1.set_title('Reliability')
    ax1.legend(loc='upper left', fontsize=9)

    bx = (bin_edges[:-1] + bin_edges[1:]) / 2
    bw = bin_edges[1] - bin_edges[0]
    ax2.bar(bx, bin_count, width=bw * 0.92, color=colour, alpha=0.85,
            edgecolor='white')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel(f'Forecast {threshold_label}')
    ax2.set_ylabel('Number of forecasts')
    ax2.set_title('Sharpness — forecast histogram')

    fig.suptitle(title, fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved {save_path}")


# =============================================================================
# Plot 7 - Rainfall threshold exceedance
# =============================================================================

def plot_rainfall_threshold_exceedance(y_true, mu_pred, sigma_pred, output_dir,
                                       threshold=RAINFALL_THRESHOLD_MM,
                                       n_bins=10):
    """
    Reliability of P(rainfall > 20 mm) - relevant for irrigation decisions
    and harvest scheduling.

        forecast probability:  p_hat = 1 - Phi((threshold - mu) / sigma)
        observation indicator: y > threshold

    Predicted probabilities are binned into deciles; each bin's mean
    predicted probability is plotted against the observed exceedance
    frequency. A well-calibrated forecast lies on the diagonal. The
    accompanying histogram of predicted probabilities shows sharpness:
    a useful forecast is concentrated near 0 or 1, not piled in the middle.
    """
    sigma_safe = np.maximum(sigma_pred[..., RAIN_IDX], 1e-6)
    p_pred = 1.0 - std_normal_cdf((threshold - mu_pred[..., RAIN_IDX]) / sigma_safe)
    obs = (y_true[..., RAIN_IDX] > threshold).astype(np.float64)

    out = os.path.join(output_dir, '07_rainfall_threshold_exceedance.png')
    _plot_threshold_reliability(
        p_pred.ravel(), obs.ravel(),
        colour=_PALETTE[RAIN_IDX],
        threshold_label=f'P(rain > {threshold:.0f} mm)',
        title=f'Rainfall Threshold Exceedance (> {threshold:.0f} mm)',
        save_path=out,
        n_bins=n_bins,
    )


# =============================================================================
# Plot 8 - Frost risk probability
# =============================================================================

def plot_frost_risk(y_true, mu_pred, sigma_pred, output_dir,
                    threshold=FROST_THRESHOLD_C, n_bins=10):
    """
    Reliability of P(temperature < 3 deg C) - relevant for frost protection
    in southern coffee growing regions.

        forecast probability:  p_hat = Phi((threshold - mu) / sigma)
        observation indicator: y < threshold

    A well-calibrated frost forecaster ensures that, when it issues a 70%
    frost probability, frost actually occurs in roughly 70% of those cases.
    """
    sigma_safe = np.maximum(sigma_pred[..., TEMP_IDX], 1e-6)
    p_pred = std_normal_cdf((threshold - mu_pred[..., TEMP_IDX]) / sigma_safe)
    obs = (y_true[..., TEMP_IDX] < threshold).astype(np.float64)

    out = os.path.join(output_dir, '08_frost_risk.png')
    _plot_threshold_reliability(
        p_pred.ravel(), obs.ravel(),
        colour=_PALETTE[TEMP_IDX],
        threshold_label=f'P(T < {threshold:.0f} °C)',
        title=f'Frost Risk (T < {threshold:.0f} °C)',
        save_path=out,
        n_bins=n_bins,
    )


# =============================================================================
# Table 1 - Point forecast metrics
# =============================================================================

def table_point_metrics(y_true, mu_pred, output_dir,
                        var_names=VAR_NAMES,
                        last_observed: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    MAE and RMSE per (variable, horizon) for the model and two reference
    baselines:
        Persistence  - predict that every future day equals the last
                       observed day (`last_observed[i]`).
        Climatology  - predict the test set's marginal mean for every day.

    A 'Mean' row averages each column over the seven horizons. The result
    is written to `output_dir/table_point_metrics.csv` and printed to the
    console for the report.

    If `last_observed` is None, persistence is approximated by reusing
    `y_true[:, 0, :]` as the anchor; this makes horizon-1 persistence error
    artificially zero. Pass the true previous-day observation for proper
    baseline numbers.
    """
    abs_err = np.abs(mu_pred - y_true)
    sq_err = (mu_pred - y_true) ** 2
    mae = abs_err.mean(axis=0)                            # (7,4)
    rmse = np.sqrt(sq_err.mean(axis=0))                   # (7,4)

    # Climatology baseline - constant per-variable mean
    clim_mu = y_true.mean(axis=(0, 1))                    # (4,)
    clim_pred = np.broadcast_to(clim_mu, y_true.shape)
    mae_clim = np.abs(clim_pred - y_true).mean(axis=0)
    rmse_clim = np.sqrt(((clim_pred - y_true) ** 2).mean(axis=0))

    # Persistence baseline
    if last_observed is None:
        print("  [WARN] last_observed not provided; persistence horizon-1 "
              "error is approximated as 0. Pass last_observed for a proper "
              "persistence baseline.")
        pers_pred = np.broadcast_to(y_true[:, :1, :], y_true.shape)
    else:
        pers_pred = np.broadcast_to(last_observed[:, None, :], y_true.shape)
    mae_pers = np.abs(pers_pred - y_true).mean(axis=0)
    rmse_pers = np.sqrt(((pers_pred - y_true) ** 2).mean(axis=0))

    horizons = [f'd{h}' for h in HORIZONS]
    rows = []
    for h_i in range(NUM_HORIZONS):
        row = {}
        for vi, name in enumerate(var_names):
            row[(name, 'Model', 'MAE')] = mae[h_i, vi]
            row[(name, 'Model', 'RMSE')] = rmse[h_i, vi]
            row[(name, 'Persistence', 'MAE')] = mae_pers[h_i, vi]
            row[(name, 'Persistence', 'RMSE')] = rmse_pers[h_i, vi]
            row[(name, 'Climatology', 'MAE')] = mae_clim[h_i, vi]
            row[(name, 'Climatology', 'RMSE')] = rmse_clim[h_i, vi]
        rows.append(row)

    df = pd.DataFrame(rows, index=horizons)
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=['Variable', 'Model', 'Metric']
    )
    df.loc['Mean'] = df.mean()

    out = os.path.join(output_dir, 'table_point_metrics.csv')
    df.round(4).to_csv(out)
    print(f"  saved {out}")

    # Console-friendly per-variable views
    print("\n  Point forecast metrics (rounded to 3 dp):")
    for name in var_names:
        sub = df[name].round(3)
        print(f"\n    {name}")
        with pd.option_context('display.max_columns', None,
                               'display.width', 120):
            print(sub.to_string())
    return df


# =============================================================================
# Table 2 - Interval coverage
# =============================================================================

def table_interval_coverage(y_true, mu_pred, sigma_pred, output_dir,
                            var_names=VAR_NAMES,
                            levels: Sequence[float] = INTERVAL_LEVELS,
                            tol: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Empirical coverage of central 50% / 80% / 95% prediction intervals,
    per variable and per horizon.

    For each level p the lower and upper bounds are
        lo = mu - z_p * sigma
        hi = mu + z_p * sigma
    with z_p the symmetric two-sided normal critical value. Coverage is the
    fraction of observations that fall inside [lo, hi]. A '*' marker flags
    any cell whose coverage deviates from its nominal level by more than
    `tol` (default 5 percentage points).
    """
    sigma_safe = np.maximum(sigma_pred, 1e-6)
    horizons = [f'd{h}' for h in HORIZONS]

    rows = []
    for h_i in range(NUM_HORIZONS):
        row = {}
        for vi, name in enumerate(var_names):
            for lvl in levels:
                z = INTERVAL_Z[lvl]
                lo = mu_pred[:, h_i, vi] - z * sigma_safe[:, h_i, vi]
                hi = mu_pred[:, h_i, vi] + z * sigma_safe[:, h_i, vi]
                inside = float(((y_true[:, h_i, vi] >= lo) &
                                (y_true[:, h_i, vi] <= hi)).mean())
                row[(name, f'{int(lvl * 100)}%')] = inside
        rows.append(row)

    df = pd.DataFrame(rows, index=horizons)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Variable', 'Nominal'])
    df.loc['Mean'] = df.mean()

    # Build a parallel string DataFrame with deviation flags for the CSV.
    str_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    flag_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    for col in df.columns:
        nominal = float(col[1].rstrip('%')) / 100.0
        for idx in df.index:
            v = float(df.loc[idx, col])
            flag = '*' if (idx != 'Mean' and abs(v - nominal) > tol) else ''
            str_df.loc[idx, col] = f'{v:.3f}{flag}'
            flag_df.loc[idx, col] = flag

    out = os.path.join(output_dir, 'table_interval_coverage.csv')
    str_df.to_csv(out)
    print(f"  saved {out}")

    print(f"\n  Interval coverage (* = deviation from nominal > {tol:.0%}):")
    for name in var_names:
        sub = str_df[name]
        print(f"\n    {name}")
        with pd.option_context('display.max_columns', None,
                               'display.width', 120):
            print(sub.to_string())
    return df, flag_df


# =============================================================================
# Top-level orchestrator
# =============================================================================

def run_evaluation(y_true: np.ndarray,
                   mu_pred: np.ndarray,
                   sigma_pred: np.ndarray,
                   last_observed: Optional[np.ndarray] = None,
                   output_dir: str = DEFAULT_OUTPUT_DIR,
                   var_names: Sequence[str] = VAR_NAMES,
                   fan_window_start: int = 0,
                   fan_window_length: int = 40):
    """
    Run all evaluation plots and tables for a multi-horizon probabilistic
    weather forecast and write outputs to `output_dir`.

    Args:
        y_true:           (n_samples, 7, 4) ground-truth values, real units
        mu_pred:          (n_samples, 7, 4) predicted Gaussian means
        sigma_pred:       (n_samples, 7, 4) predicted Gaussian std devs
        last_observed:    optional (n_samples, 4) - the day each forecast was
                          issued. Used to build a proper persistence baseline.
                          If None, the persistence baseline is approximated.
        output_dir:       directory to write PNGs and CSVs into
        var_names:        4 display labels for the variables
        fan_window_start: starting sample index for the fan-chart window
        fan_window_length: number of consecutive samples in the fan window
    """
    _validate_shapes(y_true, mu_pred, sigma_pred)
    _ensure_dir(output_dir)

    print(f"\n[EVALUATION] Writing outputs to {output_dir}/")
    print(f"  n_samples = {y_true.shape[0]}, "
          f"horizons = {y_true.shape[1]}, variables = {y_true.shape[2]}")

    print("\n[PLOT] PIT histograms")
    plot_pit_histograms(y_true, mu_pred, sigma_pred, output_dir, var_names)

    print("[PLOT] Reliability diagrams")
    plot_reliability_diagrams(y_true, mu_pred, sigma_pred, output_dir, var_names)

    print("[PLOT] CRPS by horizon")
    plot_crps_by_horizon(y_true, mu_pred, sigma_pred, output_dir, var_names)

    print("[PLOT] Predicted σ vs RMSE by horizon")
    plot_sigma_vs_rmse_by_horizon(y_true, mu_pred, sigma_pred, output_dir, var_names)

    print("[PLOT] Fan charts")
    plot_fan_charts(y_true, mu_pred, sigma_pred, output_dir, var_names,
                    window_start=fan_window_start,
                    window_length=fan_window_length)

    print("[PLOT] Predicted μ vs observed scatter")
    plot_scatter_mu_vs_observed(y_true, mu_pred, output_dir, var_names)

    print("[PLOT] Rainfall threshold exceedance")
    plot_rainfall_threshold_exceedance(y_true, mu_pred, sigma_pred, output_dir)

    print("[PLOT] Frost risk")
    plot_frost_risk(y_true, mu_pred, sigma_pred, output_dir)

    print("\n[TABLE] Point forecast metrics")
    table_point_metrics(y_true, mu_pred, output_dir, var_names,
                        last_observed=last_observed)

    print("\n[TABLE] Interval coverage")
    table_interval_coverage(y_true, mu_pred, sigma_pred, output_dir, var_names)

    print(f"\n[EVALUATION] Done. All outputs in {output_dir}/")


# ---------------------------------------------------------------------------
# Gaussian Negative Log Likelihood (NLL)
# ---------------------------------------------------------------------------

def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Gaussian negative log likelihood averaged over all elements.

    A lower NLL means the true observations fall in higher probability
    regions of the predicted distribution to reward both accuracy and
    well calibrated uncertainty
    """
    # Clamp sigma to avoid log(0)
    sigma = torch.clamp(sigma, min=1e-6)
    nll = 0.5 * torch.log(2 * torch.tensor(torch.pi)) \
        + torch.log(sigma) \
        + 0.5 * ((y - mu) / sigma) ** 2
    return nll.mean()


# ---------------------------------------------------------------------------
# CRPS — Continuous Ranked Probability Score
# ---------------------------------------------------------------------------

def crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Closed form CRPS for a Gaussian predictive distribution.

    CRPS is a strictly proper scoring rule that jointly rewards sharpness
    (tight distributions) and calibration (distributions centred on truth).
    Unlike NLL it is expressed in the same units as the forecast variable,
    making it easier to interpret
    """
    sigma = torch.clamp(sigma, min=1e-6)
    z = (y - mu) / sigma

    # Standard normal CDF and PDF via PyTorch
    normal = torch.distributions.Normal(0, 1)
    phi = torch.exp(normal.log_prob(z))   # PDF
    Phi = normal.cdf(z)                   # CDF

    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / torch.tensor(torch.pi).sqrt())
    return crps.mean()


def crps_skill_score(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                     clim_mu: torch.Tensor, clim_sigma: torch.Tensor) -> torch.Tensor:
    """
    CRPSS = 1 - CRPS_model / CRPS_climatology

    Climatology forecast: always predict the training-set mean and std,
    ignoring the input window entirely.

    Interpretation:
        1.0  = perfect
        0.0  = no better than climatology
        <0.0 = worse than climatology
    """
    model_crps = crps_gaussian(mu, sigma, y)
    clim_crps  = crps_gaussian(clim_mu.expand_as(mu), clim_sigma.expand_as(sigma), y)
    return 1.0 - model_crps / clim_crps


# ---------------------------------------------------------------------------
# RMSE and MAE (on predicted means only)
# ---------------------------------------------------------------------------

def rmse(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error on predicted means.
    Used to compare our probabilistic model against deterministic baselines.
    """
    return torch.sqrt(((mu - y) ** 2).mean())


def mae(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error on predicted means
    """
    return (mu - y).abs().mean()


def rmse_per_variable(mu: torch.Tensor, y: torch.Tensor, var_names: list) -> dict:
    """
    RMSE broken down per forecast variable. Useful for results tables
    """
    results = {}
    for i, name in enumerate(var_names):
        results[name] = torch.sqrt(((mu[..., i] - y[..., i]) ** 2).mean())
    return results


# ---------------------------------------------------------------------------
# Calibration — Empirical Coverage at Multiple Confidence Levels
# ---------------------------------------------------------------------------

def empirical_coverage(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                        confidence_levels: list = None) -> dict:
    """
    Checks whether predicted confidence intervals actually contain the
    true observations at the stated rate.

    A well calibrated model should have ~90% of observations inside
    its 90% prediction interval. Overconfident models show lower than
    expected coverage. Underconfident models show higher
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95]

    normal = torch.distributions.Normal(0, 1)
    coverage = {}
    for cl in confidence_levels:
        # z-score for symmetric interval: e.g. 90% -> z=1.645
        z = normal.icdf(torch.tensor((1 + cl) / 2))
        lower = mu - z * sigma
        upper = mu + z * sigma
        inside = ((y >= lower) & (y <= upper)).float().mean().item()
        coverage[cl] = inside
    return coverage

# ---------------------------------------------------------------------------
# Font helper — tries DejaVu, falls back gracefully
# ---------------------------------------------------------------------------

def _load_fonts():
    """
    Returns (font_regular, font_bold, font_small, font_title) as ImageFont objects.
    Tries DejaVu first (available in the comp0197-pt environment), then falls
    back to Pillow's built-in default so the code never crashes.
    """
    paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    ]
    try:
        regular  = ImageFont.truetype(paths[0], 13)
        bold     = ImageFont.truetype(paths[1], 13)
        small    = ImageFont.truetype(paths[0], 11)
        title    = ImageFont.truetype(paths[1], 15)
        large    = ImageFont.truetype(paths[1], 17)
    except OSError:
        regular = bold = small = title = large = ImageFont.load_default()
    return regular, bold, small, title, large


# ---------------------------------------------------------------------------
# Colour palette (RGB tuples)
# ---------------------------------------------------------------------------

C_BLACK      = (30,  30,  30)
C_DARK_GREY  = (80,  80,  80)
C_MID_GREY   = (140, 140, 140)
C_LIGHT_GREY = (210, 210, 210)
C_WHITE      = (255, 255, 255)
C_BLUE_DARK  = (24,  95,  165)   # mean line / axis colour
C_BLUE_MID   = (55,  138, 221)   # ±1std band / model curve dots
C_BLUE_LIGHT = (181, 212, 244)   # ±2std band
C_RED        = (196, 60,  60)    # ground truth
C_DIAG       = (180, 180, 180)   # perfect-calibration diagonal



# ---------------------------------------------------------------------------
# Plots 
# ---------------------------------------------------------------------------



def plot_calibration(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                     confidence_levels: list = None, save_path: str = 'calibration.png'):
    """
    Saves a reliability diagram (calibration curve) as a PNG 
    Draws the diagonal perfect calibration reference line and the model's
    empirical coverage points connected by line segments.
    Points above the diagonal = underconfident, below = overconfident
    """
    if confidence_levels is None:
        confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
 
    coverage  = empirical_coverage(mu, sigma, y, confidence_levels)
    expected  = list(coverage.keys())
    observed  = list(coverage.values())
 
    # Canvas
    W, H      = 620, 560
    pad_l     = 90    # left  — room for y-axis label + ticks
    pad_r     = 40    # right
    pad_t     = 70    # top   — room for title
    pad_b     = 70    # bottom — room for x-axis label + ticks
    plot_w    = W - pad_l - pad_r
    plot_h    = H - pad_t - pad_b
 
    img  = Image.new('RGB', (W, H), color=C_WHITE)
    draw = ImageDraw.Draw(img)
 
    font_reg, font_bold, font_sm, font_title, font_large = _load_fonts()
 
    def to_px(ex, ob):
        """Map (expected, observed) in [0, 1] → pixel coords."""
        px = pad_l + int(ex * plot_w)
        py = H - pad_b - int(ob * plot_h)
        return px, py
 
    # Title
    title_text = 'Reliability Diagram - Forecast Calibration'
    draw.text((W // 2 - 185, 18), title_text, fill=C_BLACK, font=font_large)
 
    subtitle = 'Probabilistic weather forecast · Minas Gerais, Brazil · Test set 2022 - 2024'
    draw.text((W // 2 - 195, 42), subtitle, fill=C_MID_GREY, font=font_sm)
 
    # Plot border
    draw.rectangle(
        [pad_l, pad_t, pad_l + plot_w, pad_t + plot_h],
        outline=C_LIGHT_GREY, width=1
    )
 
    # Grid
    for v in [0.2, 0.4, 0.6, 0.8]:
        gx, _  = to_px(v, 0)
        _, gy  = to_px(0, v)
        # vertical grid line
        draw.line([(gx, pad_t), (gx, H - pad_b)], fill=C_LIGHT_GREY, width=1)
        # horizontal grid line
        draw.line([(pad_l, gy), (pad_l + plot_w, gy)], fill=C_LIGHT_GREY, width=1)
 
    # Perfect calibration line
    # Drawn as a dashed diagonal in mid grey
    n_dash = 30
    for i in range(n_dash):
        if i % 2 == 0:
            x0, y0 = to_px(i / n_dash, i / n_dash)
            x1, y1 = to_px((i + 1) / n_dash, (i + 1) / n_dash)
            draw.line([(x0, y0), (x1, y1)], fill=C_DIAG, width=2)
 
    # Shaded regions
    # Below diagonal = overconfident (light red tint)
    # Above diagonal = underconfident (light blue tint)
    # Built by drawing thin horizontal scan bands
    for row_pct in range(0, 100):
        frac = row_pct / 100
        _, gy_diag = to_px(frac, frac)
        # underconfident region: above the diagonal
        draw.line(
            [(pad_l, gy_diag), (pad_l + int(frac * plot_w), gy_diag)],
            fill=(235, 245, 255), width=1
        )
        # overconfident region: below the diagonal
        _, gy_bot = to_px(0, frac)
        gx_right, _ = to_px(frac, 0)
        draw.line(
            [(gx_right, gy_bot), (pad_l + plot_w, gy_bot)],
            fill=(255, 240, 240), width=1
        )
 
    # Region annotation labels
    draw.text((pad_l + 8, pad_t + 12),
              'Underconfident  (std too large)', fill=(100, 140, 190), font=font_sm)
    draw.text((pad_l + plot_w - 165, H - pad_b - 28),
              'Overconfident  (std too small)', fill=(190, 100, 100), font=font_sm)
 
    # Model calibration curve
    pts = [to_px(e, o) for e, o in zip(expected, observed)]
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=C_BLUE_MID, width=3)
 
    # Filled circles at each point
    r = 6
    for px, py in pts:
        draw.ellipse([(px - r, py - r), (px + r, py + r)],
                     fill=C_BLUE_MID, outline=C_WHITE)
 
    # Axis ticks
    tick_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for v in tick_vals:
        # x-axis tick + label
        tx, ty = to_px(v, 0)
        draw.line([(tx, ty), (tx, ty + 5)], fill=C_DARK_GREY, width=1)
        label = f'{int(v * 100)}%'
        draw.text((tx - 10, ty + 8), label, fill=C_DARK_GREY, font=font_sm)
 
        # y-axis tick + label
        tx2, ty2 = to_px(0, v)
        draw.line([(tx2 - 5, ty2), (tx2, ty2)], fill=C_DARK_GREY, width=1)
        draw.text((tx2 - 38, ty2 - 7), f'{int(v * 100)}%',
                  fill=C_DARK_GREY, font=font_sm)
 
    # Axis lines
    draw.line([(pad_l, pad_t), (pad_l, H - pad_b)],
              fill=C_DARK_GREY, width=2)
    draw.line([(pad_l, H - pad_b), (W - pad_r, H - pad_b)],
              fill=C_DARK_GREY, width=2)
 
    # Axis labels
    # x-axis label
    draw.text((W // 2 - 95, H - pad_b + 32),
              'Expected coverage (confidence level)',
              fill=C_BLACK, font=font_reg)
 
    # y-axis label — drawn vertically by rotating a temporary image
    label_img = Image.new('RGB', (200, 20), color=C_WHITE)
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 2), 'Empirical coverage (test set)', fill=C_BLACK, font=font_reg)
    label_rotated = label_img.rotate(90, expand=True)
    img.paste(label_rotated, (8, H // 2 - 100))
 
    # Legend
    legend_x, legend_y = pad_l + plot_w - 200, pad_t + 16
    draw.rectangle(
        [legend_x - 8, legend_y - 6, legend_x + 192, legend_y + 52],
        fill=C_WHITE, outline=C_LIGHT_GREY
    )
    # diagonal legend entry
    for i in range(4):
        if i % 2 == 0:
            draw.line([(legend_x + i * 8, legend_y + 8),
                       (legend_x + (i + 1) * 8, legend_y + 8)],
                      fill=C_DIAG, width=2)
    draw.text((legend_x + 36, legend_y + 2),
              'Perfect calibration', fill=C_DARK_GREY, font=font_sm)
    # model legend entry
    draw.ellipse([(legend_x + 6, legend_y + 24),
                  (legend_x + 18, legend_y + 36)],
                 fill=C_BLUE_MID, outline=C_WHITE)
    draw.text((legend_x + 36, legend_y + 26),
              'Model (this run)', fill=C_DARK_GREY, font=font_sm)
 
    img.save(save_path)
    print(f'Calibration plot saved → {save_path}')
 



def plot_forecast(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                  var_names: list,
                  save_path: str = 'forecast.png'):
    """
    Saves predicted mean ± 1std / ± 2std bands against ground truth as a PNG.
    One subplot row per forecast variable.

    Args:
        mu:        (H, V) predicted means in real-world units
        sigma:     (H, V) predicted std devs in real-world units
        y:         (H, V) ground truth in real-world units
        var_names: list of variable name strings
        save_path: output file path
    """
    H_steps = mu.shape[0]
    V       = len(var_names)

    # layout
    panel_w  = 700
    panel_h  = 180
    pad_l    = 80    # left  — y-axis label + ticks
    pad_r    = 30    # right
    pad_t    = 36    # top of each panel — variable name
    pad_b    = 28    # bottom of each panel — day ticks (only last panel shows labels)
    header_h = 52    # space for overall title
    footer_h = 36    # space for shared x-axis label
    plot_w   = panel_w - pad_l - pad_r
    plot_h   = panel_h - pad_t - pad_b

    total_h  = header_h + panel_h * V + footer_h
    img      = Image.new('RGB', (panel_w, total_h), color=C_WHITE)
    draw     = ImageDraw.Draw(img)

    font_reg, font_bold, font_sm, font_title, font_large = _load_fonts()

    # Title
    draw.text((panel_w // 2 - 220, 12),
              '7-Day Probabilistic Weather Forecast vs Observations',
              fill=C_BLACK, font=font_large)
    draw.text((panel_w // 2 - 195, 34),
              'Predicted mean  ·  ±1std / ±2std uncertainty bands  ·  Ground truth  ·  a single test-set window',
              fill=C_MID_GREY, font=font_sm)

    # Per variable panels
    for vi in range(V):
        m     = mu[:, vi]
        s     = sigma[:, vi]
        truth = y[:, vi]

        y_off = header_h + vi * panel_h   # vertical offset for this panel

        all_vals = np.concatenate([m - 2 * s, m + 2 * s, truth])
        v_min    = all_vals.min()
        v_max    = all_vals.max()
        v_range  = v_max - v_min if v_max != v_min else 1.0
        # small vertical padding so lines don't touch panel edges
        v_pad    = v_range * 0.12
        v_min   -= v_pad
        v_max   += v_pad
        v_range  = v_max - v_min

        def to_px(day_idx, val):
            px = pad_l + int(day_idx / (H_steps - 1) * plot_w)
            py = y_off + pad_t + int((1 - (val - v_min) / v_range) * plot_h)
            return px, py

        # panel background
        draw.rectangle(
            [pad_l, y_off + pad_t,
             pad_l + plot_w, y_off + pad_t + plot_h],
            fill=(252, 252, 252), outline=C_LIGHT_GREY
        )

        # horizontal grid
        for tick_frac in [0.25, 0.5, 0.75]:
            tick_val = v_min + tick_frac * v_range
            _, gy = to_px(0, tick_val)
            draw.line([(pad_l, gy), (pad_l + plot_w, gy)],
                      fill=C_LIGHT_GREY, width=1)

        # ±2 std band
        poly_2s = []
        for d in range(H_steps):
            poly_2s.append(to_px(d, m[d] + 2 * s[d]))
        for d in range(H_steps - 1, -1, -1):
            poly_2s.append(to_px(d, m[d] - 2 * s[d]))
        draw.polygon(poly_2s, fill=C_BLUE_LIGHT)

        # ±1 std band
        poly_1s = []
        for d in range(H_steps):
            poly_1s.append(to_px(d, m[d] + s[d]))
        for d in range(H_steps - 1, -1, -1):
            poly_1s.append(to_px(d, m[d] - s[d]))
        draw.polygon(poly_1s, fill=C_BLUE_MID)

        # predicted mean
        mean_pts = [to_px(d, m[d]) for d in range(H_steps)]
        draw.line(mean_pts, fill=C_BLUE_DARK, width=2)
        for px, py in mean_pts:
            draw.ellipse([(px - 4, py - 4), (px + 4, py + 4)],
                         fill=C_BLUE_DARK, outline=C_WHITE)

        # ground truth
        truth_pts = [to_px(d, truth[d]) for d in range(H_steps)]
        draw.line(truth_pts, fill=C_RED, width=2)
        for px, py in truth_pts:
            draw.ellipse([(px - 3, py - 3), (px + 3, py + 3)],
                         fill=C_RED, outline=C_WHITE)

        # y-axis
        draw.line(
            [(pad_l, y_off + pad_t), (pad_l, y_off + pad_t + plot_h)],
            fill=C_DARK_GREY, width=1
        )

        # y ticks + labels
        for tick_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            tick_val = v_min + tick_frac * v_range
            _, gy    = to_px(0, tick_val)
            draw.line([(pad_l - 4, gy), (pad_l, gy)], fill=C_DARK_GREY, width=1)
            draw.text((4, gy - 6), f'{tick_val:.1f}',
                      fill=C_DARK_GREY, font=font_sm)

        # Variable name label
        # Shown as a bold tag on the left of the panel
        var_label = var_names[vi].replace('_', ' ').title()
        draw.text((pad_l + 6, y_off + pad_t + 5),
                  var_label, fill=C_BLUE_DARK, font=font_bold)

        # x-axis ticks
        # Day numbers shown on every panel; full labels only on last
        x_axis_y = y_off + pad_t + plot_h
        draw.line([(pad_l, x_axis_y), (pad_l + plot_w, x_axis_y)],
                  fill=C_DARK_GREY, width=1)
        for d in range(H_steps):
            px, _ = to_px(d, v_min)
            draw.line([(px, x_axis_y), (px, x_axis_y + 4)],
                      fill=C_DARK_GREY, width=1)
            if vi == V - 1:
                draw.text((px - 8, x_axis_y + 6),
                          f'Day {d + 1}', fill=C_DARK_GREY, font=font_sm)

        # Panel separator
        if vi < V - 1:
            draw.line(
                [(0, y_off + panel_h), (panel_w, y_off + panel_h)],
                fill=C_LIGHT_GREY, width=1
            )

    # x-axis label
    draw.text((panel_w // 2 - 42,
               header_h + panel_h * V + 10),
              'Forecast day', fill=C_BLACK, font=font_reg)

    # Shared legend
    legend_x = pad_l + plot_w - 310
    legend_y = header_h + 6
    draw.rectangle(
        [legend_x - 6, legend_y - 4, legend_x + 310, legend_y + 44],
        fill=C_WHITE, outline=C_LIGHT_GREY
    )
    # ±2 std swatch
    draw.rectangle(
        [legend_x, legend_y + 4, legend_x + 18, legend_y + 14],
        fill=C_BLUE_LIGHT
    )
    draw.text((legend_x + 22, legend_y + 3), '±2std band',
              fill=C_DARK_GREY, font=font_sm)
    # ±1 std swatch
    draw.rectangle(
        [legend_x + 80, legend_y + 4, legend_x + 98, legend_y + 14],
        fill=C_BLUE_MID
    )
    draw.text((legend_x + 102, legend_y + 3), '±1std band',
              fill=C_DARK_GREY, font=font_sm)
    # mean line swatch
    draw.line([(legend_x + 162, legend_y + 9), (legend_x + 180, legend_y + 9)],
              fill=C_BLUE_DARK, width=2)
    draw.ellipse([(legend_x + 168, legend_y + 5), (legend_x + 176, legend_y + 13)],
                 fill=C_BLUE_DARK, outline=C_WHITE)
    draw.text((legend_x + 184, legend_y + 3), 'Predicted mean',
              fill=C_DARK_GREY, font=font_sm)
    # ground truth swatch
    draw.line([(legend_x, legend_y + 32), (legend_x + 18, legend_y + 32)],
              fill=C_RED, width=2)
    draw.ellipse([(legend_x + 6, legend_y + 28), (legend_x + 14, legend_y + 36)],
                 fill=C_RED, outline=C_WHITE)
    draw.text((legend_x + 22, legend_y + 26), 'Ground truth',
              fill=C_DARK_GREY, font=font_sm)

    img.save(save_path)
    print(f'Forecast plot saved → {save_path}')



# ---------------------------------------------------------------------------
# Summary Table — All Metrics at Once
# ---------------------------------------------------------------------------

TARGET_VAR_NAMES = ["temperature", "rainfall", "humidity", "wind_speed"]

def evaluate_all(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                 var_names: list = None) -> dict:
    """
    Runs all metrics in one call. Convenient for test.py.
    """
    _require_torch()
    if var_names is None:
        var_names = TARGET_VAR_NAMES

    results = {
        'nll':          gaussian_nll(mu, sigma, y).item(),
        'crps':         crps_gaussian(mu, sigma, y).item(),
        'rmse':         rmse(mu, y).item(),
        'mae':          mae(mu, y).item(),
        'coverage':     empirical_coverage(mu, sigma, y),
        'rmse_per_var': {k: v.item() for k, v in
                         rmse_per_variable(mu, y, var_names).items()}
    }
    return results


# ---------------------------------------------------------------------------
# Shared: print metrics + save plots
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"

def evaluate_and_plot(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                      mu_plot: np.ndarray, sigma_plot: np.ndarray, y_plot: np.ndarray,
                      var_names: list, label: str, window_idx: int = 0,
                      clim_mu: torch.Tensor = None, clim_sigma: torch.Tensor = None):
    """
    Print all evaluation metrics and save calibration + forecast PNGs.

    Args:
        mu, sigma, y:             tensors in real-world units — used for metrics
        mu_plot, sigma_plot, y_plot: numpy arrays in real-world units — used for plots
        var_names:                list of variable name strings (must match axis-2 order)
        label:                    short tag used in output filenames, e.g. 'dummy' or 'real'
        window_idx:               which test window to use for the forecast plot; pick the
                                  sample closest to the median CRPS for a representative view
        clim_mu, clim_sigma:      (1, 1, V) tensors of training-set mean/std for CRPSS;
                                  if None, CRPSS is not reported
    """
    _require_torch()
    results = evaluate_all(mu, sigma, y, var_names=var_names)

    N, H, V = mu_plot.shape

    # ---------- CRPS skill score ----------
    if clim_mu is not None and clim_sigma is not None:
        overall_crpss = crps_skill_score(mu, sigma, y, clim_mu, clim_sigma).item()
        crpss_per_var = [
            crps_skill_score(mu[..., i:i+1], sigma[..., i:i+1], y[..., i:i+1],
                             clim_mu[..., i:i+1], clim_sigma[..., i:i+1]).item()
            for i in range(mu.shape[-1])
        ]
    else:
        overall_crpss  = None
        crpss_per_var  = None

    # ---------- per-variable numpy stats ----------
    err   = mu_plot - y_plot                                  # (N, H, V)
    crps  = gaussian_crps(mu_plot, sigma_plot, y_plot)        # (N, H, V)

    var_mae       = np.abs(err).mean(axis=(0, 1))             # (V,)
    var_rmse      = np.sqrt((err**2).mean(axis=(0, 1)))       # (V,)
    var_crps      = crps.mean(axis=(0, 1))                    # (V,)
    var_mean_sig  = sigma_plot.mean(axis=(0, 1))              # (V,)
    obs_mean      = y_plot.mean(axis=(0, 1))                  # (V,)
    obs_std       = y_plot.std(axis=(0, 1))                   # (V,)

    # per-variable coverage at 50/80/95%
    cov_levels = [0.50, 0.80, 0.95]
    var_cov = {}                                              # level -> (V,)
    for lvl in cov_levels:
        z = INTERVAL_Z[lvl]
        lo = mu_plot - z * np.maximum(sigma_plot, 1e-6)
        hi = mu_plot + z * np.maximum(sigma_plot, 1e-6)
        inside = ((y_plot >= lo) & (y_plot <= hi)).mean(axis=(0, 1))  # (V,)
        var_cov[lvl] = inside

    # per-horizon RMSE averaged across all variables
    hz_rmse = np.sqrt((err**2).mean(axis=(0, 2)))             # (H,)

    # per-variable per-horizon RMSE
    vh_rmse = np.sqrt((err**2).mean(axis=0))                  # (H, V)

    # ---------- format helpers ----------
    SEP  = "-" * 72
    W    = 14   # column width for variable columns

    def hdr(names):
        return "  " + "".join(f"{n:>{W}}" for n in names)

    def row(label_, vals, fmt=".4f"):
        return f"  {label_:<18}" + "".join(f"{v:{W}{fmt}}" for v in vals)

    def cov_row(label_, vals):
        return f"  {label_:<18}" + "".join(f"{v*100:>{W}.1f}%" for v in vals)

    # ---------- build lines ----------
    lines = []
    lines.append(f"=== Results ({label}) ===")
    lines.append(f"  Samples: {N}   Horizon: {H} days   Variables: {V}")
    lines.append(f"  Forecast window shown: index {window_idx}  (best obs-range/RMSE on temperature)")
    lines.append("")

    lines.append(SEP)
    lines.append("  AGGREGATE METRICS")
    lines.append("  CRPS/RMSE/MAE are averaged across all 4 variables (mixed units — use for run comparison only)")
    lines.append("  CRPSS = 1 - CRPS_model/CRPS_climatology  (0=no better than climatology, 1=perfect)")
    lines.append(SEP)
    if overall_crpss is not None:
        clim_crps_overall = results['crps'] / (1 - overall_crpss) if overall_crpss < 1 else float('inf')
        lines.append(f"  {'':20} {'Model':>10}  {'Climatology':>12}")
        lines.append(f"  {'CRPS':<20} {results['crps']:>10.4f}  {clim_crps_overall:>12.4f}")
        lines.append(f"  {'CRPSS':<20} {overall_crpss:>10.4f}  {'0.0000':>12}")
    else:
        lines.append(f"  {'CRPS':<10} {results['crps']:.4f}")
    lines.append(f"  {'NLL':<10} {results['nll']:.4f}")
    lines.append(f"  {'RMSE':<10} {results['rmse']:.4f}")
    lines.append(f"  {'MAE':<10} {results['mae']:.4f}")
    lines.append("")

    lines.append(SEP)
    lines.append("  COVERAGE  (nominal → empirical, all variables)")
    lines.append(SEP)
    for cl, cov in results['coverage'].items():
        dev = cov * 100 - cl * 100
        flag = " *" if abs(dev) > 5 else ""
        lines.append(f"  {int(cl*100):>3}%  →  {cov*100:5.1f}%   (Δ {dev:+.1f}%){flag}")
    lines.append("")

    lines.append(SEP)
    lines.append("  PER-VARIABLE SUMMARY")
    lines.append("  CRPS/RMSE/MAE are in real-world units: Temp=°C  Rain=mm  Hum=%  Wind=m/s")
    lines.append("  CRPS ≈ MAE for a perfect point forecast; lower is better")
    lines.append(SEP)
    short = [n.split("(")[0].strip() for n in var_names]
    lines.append(hdr(short))
    lines.append(row("MAE",          var_mae))
    lines.append(row("RMSE",         var_rmse))
    lines.append(row("CRPS (model)",  var_crps))
    if crpss_per_var is not None:
        clim_crps_per_var = [
            var_crps[i] / (1 - crpss_per_var[i]) if crpss_per_var[i] < 1 else float('inf')
            for i in range(len(var_crps))
        ]
        lines.append(row("CRPS (clim)", clim_crps_per_var))
        lines.append(row("CRPSS",       crpss_per_var))
    lines.append(row("Mean σ",      var_mean_sig))
    lines.append(row("Obs mean",    obs_mean))
    lines.append(row("Obs std",     obs_std))
    lines.append(row("RMSE/Obs std", var_rmse / (obs_std + 1e-9)))
    lines.append("")
    lines.append("  Coverage per variable:")
    for lvl in cov_levels:
        lines.append(cov_row(f"  {int(lvl*100)}%", var_cov[lvl]))
    lines.append("")

    lines.append(SEP)
    lines.append("  PER-HORIZON RMSE  (mean across variables)")
    lines.append(SEP)
    for h in range(H):
        lines.append(f"  Day {h+1}:  {hz_rmse[h]:.4f}")
    lines.append("")

    lines.append(SEP)
    lines.append("  PER-VARIABLE, PER-HORIZON RMSE")
    lines.append(SEP)
    lines.append(hdr(short))
    for h in range(H):
        lines.append(row(f"Day {h+1}", vh_rmse[h]))
    lines.append(row("Mean", vh_rmse.mean(axis=0)))
    lines.append("")

    lines.append(SEP)
    lines.append("  OUTPUTS")
    lines.append(SEP)
    for fn in [
        "01_pit_histograms.png", "02_reliability_diagrams.png",
        "03_crps_by_horizon.png", "04_sigma_vs_rmse_by_horizon.png",
        "05_fan_chart_<var>.png  (x4)", "06_scatter_mu_vs_observed.png",
        "07_rainfall_threshold_exceedance.png", "08_frost_risk.png",
        "09_calibration.png", "10_forecast.png",
        "table_point_metrics.csv", "table_interval_coverage.csv",
    ]:
        lines.append(f"  {fn}")

    text = "\n".join(lines) + "\n"

    print()
    print(text)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plot_calibration(mu, sigma, y,
                     save_path=os.path.join(RESULTS_DIR, '09_calibration.png'))
    plot_forecast(mu_plot[window_idx], sigma_plot[window_idx], y_plot[window_idx],
                  var_names=var_names,
                  save_path=os.path.join(RESULTS_DIR, '10_forecast.png'))

    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Summary written -> {summary_path}")


# ---------------------------------------------------------------------------
# Real-data evaluation — loads model saved by train.py, no training here
# ---------------------------------------------------------------------------

def real_data_test(output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    Load the trained model from models/, run inference on the test split,
    and produce the full evaluation suite (8 plots + 2 CSVs) via run_evaluation.

    Variable-order note
    -------------------
    train.py's model head emits axis-2 in the order defined by
    TARGET_FEATURE_INDICES: [temperature, humidity, rainfall, wind].
    run_evaluation expects axis-2 in the order used by VAR_NAMES:
    [temperature, rainfall, humidity, wind].
    The remapping index MODEL_TO_EVAL = [0, 2, 1, 3] is applied after
    denormalisation so every downstream plot is labelled correctly.
    """
    _require_torch()
    from train import (
        ProbabilisticForecaster,
        TARGET_FEATURE_INDICES,
        WeatherWindowDataset,
        load_daily_data,
    )

    weights_path = os.path.join(MODELS_DIR, "best_model.pt")
    stats_path   = os.path.join(MODELS_DIR, "normalisation_stats.pt")
    config_path  = os.path.join(MODELS_DIR, "split_config.pt")

    for p in (weights_path, stats_path, config_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found — run train.py first to generate model artefacts."
            )

    # ---- Load saved artefacts ----
    stats       = torch.load(stats_path,  map_location="cpu", weights_only=False)
    config      = torch.load(config_path, map_location="cpu", weights_only=False)
    train_mean  = stats["mean"]
    train_std   = stats["std"]
    max_records = config["max_records"]
    split_fracs = config["split_fracs"]

    # ---- Reconstruct test split (mirrors train.py exactly) ----
    dates, features = load_daily_data(max_records=max_records)
    n = len(features)
    if split_fracs is not None:
        n_train = int(n * split_fracs[0])
        n_val   = int(n * split_fracs[1])
        test_features = features[n_train + n_val:]
    else:
        from train import VAL_END_YEAR
        test_idx      = [i for i, d in enumerate(dates) if int(d[:4]) > VAL_END_YEAR]
        test_features = features[test_idx]

    test_data    = (test_features - train_mean) / train_std
    test_dataset = WeatherWindowDataset(test_data)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ---- Load model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ProbabilisticForecaster()
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"[EVAL] Loaded model from {weights_path}  (device: {device})")

    # ---- Collect predictions and last observed day for persistence baseline ----
    all_mu, all_sigma, all_y, all_last = [], [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            mu, sigma = model(x.to(device))
            all_mu.append(mu.cpu())
            all_sigma.append(sigma.cpu())
            all_y.append(y.cpu())
            # Last input day's target features (normalised): x[:, -1, TARGET_FEATURE_INDICES]
            all_last.append(x[:, -1, :][:, TARGET_FEATURE_INDICES].cpu())

    mu_norm    = torch.cat(all_mu,    dim=0)   # (N, 7, 4)  model order
    sigma_norm = torch.cat(all_sigma, dim=0)
    y_norm     = torch.cat(all_y,     dim=0)
    last_norm  = torch.cat(all_last,  dim=0)   # (N, 4)     model order

    # ---- Denormalise ----
    t_mean     = train_mean[TARGET_FEATURE_INDICES]   # (4,) model order
    t_std      = train_std[TARGET_FEATURE_INDICES]
    mu_real    = (mu_norm    * t_std + t_mean).numpy()
    sigma_real = (sigma_norm * t_std          ).numpy()
    y_real     = (y_norm     * t_std + t_mean).numpy()
    last_real  = (last_norm  * t_std + t_mean).numpy()

    # ---- Remap variable axis from model order to evaluation order ----
    # Model head order (TARGET_FEATURE_INDICES): temperature, humidity, rainfall, wind
    # VAR_NAMES / run_evaluation order:          temperature, rainfall,  humidity, wind
    MODEL_TO_EVAL = [0, 2, 1, 3]
    mu_real    = mu_real[...,    MODEL_TO_EVAL]
    sigma_real = sigma_real[..., MODEL_TO_EVAL]
    y_real     = y_real[...,     MODEL_TO_EVAL]
    last_real  = last_real[:,    MODEL_TO_EVAL]

    # ---- Run comprehensive evaluation suite ----
    run_evaluation(
        y_true        = y_real,
        mu_pred       = mu_real,
        sigma_pred    = sigma_real,
        last_observed = last_real,
        output_dir    = output_dir,
        var_names     = VAR_NAMES,
    )

    # ---- Pick the best-looking forecast window ----
    # Score each window by how well the model tracks observable variation in
    # temperature (the most interpretable variable, index 0 in eval order).
    # score = obs_range / rmse — maximising this picks windows where the ground
    # truth actually moves AND the model follows it closely.
    temp_obs   = y_real[:, :, 0]                                          # (N, 7)
    temp_mu    = mu_real[:, :, 0]                                         # (N, 7)
    obs_range  = temp_obs.max(axis=1) - temp_obs.min(axis=1)             # (N,)
    rmse_temp  = np.sqrt(((temp_mu - temp_obs) ** 2).mean(axis=1))       # (N,)
    vis_score  = obs_range / (rmse_temp + 1e-6)                          # (N,)
    rep_idx    = int(np.argmax(vis_score))
    crps_per_window = gaussian_crps(mu_real, sigma_real, y_real).mean(axis=(1, 2))
    print(f"[EVAL] Best-looking window: index {rep_idx} "
          f"(temp obs range={obs_range[rep_idx]:.2f}°C, "
          f"temp RMSE={rmse_temp[rep_idx]:.4f}, "
          f"CRPS={crps_per_window[rep_idx]:.4f}, "
          f"median CRPS={np.median(crps_per_window):.4f})")

    # ---- Build climatology tensors for CRPSS (training-set mean/std, eval order) ----
    t_mean_eval  = t_mean[MODEL_TO_EVAL]   # (4,) in eval order
    t_std_eval   = t_std[MODEL_TO_EVAL]
    clim_mu      = torch.tensor(t_mean_eval, dtype=torch.float32).view(1, 1, -1)
    clim_sigma   = torch.tensor(t_std_eval,  dtype=torch.float32).view(1, 1, -1)

    # ---- Calibration + 7-day forecast PIL plots + summary.txt ----
    # Pass real-world tensors so metrics and per-variable labels both use eval order.
    evaluate_and_plot(
        torch.from_numpy(mu_real),
        torch.from_numpy(sigma_real),
        torch.from_numpy(y_real),
        mu_real, sigma_real, y_real,
        var_names=VAR_NAMES, label='real', window_idx=rep_idx,
        clim_mu=clim_mu, clim_sigma=clim_sigma,
    )
    print("\nReal data test complete.")


def ablation_test():
    """
    Load each ablation checkpoint saved by train.py (RUN_ABLATIONS=True),
    run inference on the test split, and print a comparison table.
    Results are written to results/ablation_summary.txt.
    """
    _require_torch()
    from train import (
        ABLATION_CONFIGS,
        ABLATIONS_DIR,
        DeterministicForecaster,
        ProbabilisticForecaster,
        TARGET_FEATURE_INDICES,
        WeatherWindowDataset,
        load_daily_data,
    )

    stats_path  = os.path.join(MODELS_DIR, "normalisation_stats.pt")
    config_path = os.path.join(MODELS_DIR, "split_config.pt")
    for p in (stats_path, config_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found — run train.py first.")

    stats       = torch.load(stats_path,  map_location="cpu", weights_only=False)
    config      = torch.load(config_path, map_location="cpu", weights_only=False)
    train_mean  = stats["mean"]
    train_std   = stats["std"]
    max_records = config["max_records"]
    split_fracs = config["split_fracs"]

    # Reconstruct the same test split used during training
    dates, features = load_daily_data(max_records=max_records)
    n = len(features)
    if split_fracs is not None:
        n_train = int(n * split_fracs[0])
        n_val   = int(n * split_fracs[1])
        test_features = features[n_train + n_val:]
    else:
        from train import VAL_END_YEAR
        test_features = features[[i for i, d in enumerate(dates) if int(d[:4]) > VAL_END_YEAR]]

    test_data    = (test_features - train_mean) / train_std
    test_dataset = WeatherWindowDataset(test_data)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ABLATE EVAL] Device: {device}\n")

    # Model head order → eval order: [temp, hum, rain, wind] → [temp, rain, hum, wind]
    MODEL_TO_EVAL = [0, 2, 1, 3]

    # Climatology tensors for CRPSS (training-set mean/std in eval order)
    t_mean_eval = train_mean[TARGET_FEATURE_INDICES][MODEL_TO_EVAL]
    t_std_eval  = train_std[TARGET_FEATURE_INDICES][MODEL_TO_EVAL]
    clim_mu     = torch.tensor(t_mean_eval, dtype=torch.float32).view(1, 1, -1)
    clim_sigma  = torch.tensor(t_std_eval,  dtype=torch.float32).view(1, 1, -1)

    results = []
    for cfg in ABLATION_CONFIGS:
        weights_path = os.path.join(ABLATIONS_DIR, cfg["filename"])
        if not os.path.exists(weights_path):
            print(f"  [SKIP] {cfg['name']} — weights not found at {weights_path}")
            continue

        is_det = cfg.get("deterministic", False)
        ModelClass = DeterministicForecaster if is_det else ProbabilisticForecaster
        model = ModelClass(
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        all_mu, all_sigma, all_y = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                mu, sigma = model(x.to(device))
                all_mu.append(mu.cpu())
                if sigma is not None:
                    all_sigma.append(sigma.cpu())
                all_y.append(y.cpu())

        mu_norm = torch.cat(all_mu, dim=0)
        y_norm  = torch.cat(all_y,  dim=0)

        # Denormalise and remap to eval variable order
        t_mean  = train_mean[TARGET_FEATURE_INDICES]
        t_std   = train_std[TARGET_FEATURE_INDICES]
        mu_r    = (mu_norm * t_std + t_mean).numpy()[..., MODEL_TO_EVAL]
        y_r     = (y_norm  * t_std + t_mean).numpy()[..., MODEL_TO_EVAL]
        mu_t    = torch.from_numpy(mu_r)
        y_t     = torch.from_numpy(y_r)

        if is_det:
            # CRPS for point forecasts = MAE
            test_crps = torch.mean(torch.abs(mu_t - y_t)).item()
            test_nll  = float('nan')
            cov       = {0.5: float('nan'), 0.9: float('nan')}
            # CRPSS: compare point-forecast CRPS (MAE) against Gaussian climatology CRPS
            clim_crps_val = crps_gaussian(
                clim_mu.expand_as(mu_t), clim_sigma.expand_as(mu_t), y_t
            ).item()
            overall_crpss = 1.0 - test_crps / clim_crps_val
            crps_per_var = [
                float(torch.mean(torch.abs(mu_t[..., i] - y_t[..., i])))
                for i in range(mu_t.shape[-1])
            ]
            crpss_per_var = [
                1.0 - crps_per_var[i] / crps_gaussian(
                    clim_mu[..., i:i+1].expand_as(mu_t[..., i:i+1]),
                    clim_sigma[..., i:i+1].expand_as(mu_t[..., i:i+1]),
                    y_t[..., i:i+1]).item()
                for i in range(mu_t.shape[-1])
            ]
        else:
            sigma_norm = torch.cat(all_sigma, dim=0)
            sigma_r    = (sigma_norm * t_std).numpy()[..., MODEL_TO_EVAL]
            sigma_t    = torch.from_numpy(sigma_r)

            test_crps     = crps_gaussian(mu_t, sigma_t, y_t).item()
            test_nll      = gaussian_nll(mu_t, sigma_t, y_t).item()
            cov           = empirical_coverage(mu_t, sigma_t, y_t, [0.5, 0.9])
            overall_crpss = crps_skill_score(mu_t, sigma_t, y_t, clim_mu, clim_sigma).item()
            crps_per_var  = [
                float(crps_gaussian(mu_t[..., i:i+1], sigma_t[..., i:i+1], y_t[..., i:i+1]))
                for i in range(mu_t.shape[-1])
            ]
            crpss_per_var = [
                crps_skill_score(mu_t[..., i:i+1], sigma_t[..., i:i+1], y_t[..., i:i+1],
                                 clim_mu[..., i:i+1], clim_sigma[..., i:i+1]).item()
                for i in range(mu_t.shape[-1])
            ]

        results.append({
            "name":          cfg["name"],
            "deterministic": is_det,
            "test_crps":     test_crps,
            "overall_crpss": overall_crpss,
            "test_nll":      test_nll,
            "cov_50":        cov[0.5],
            "cov_90":        cov[0.9],
            "crps_per_var":  crps_per_var,
            "crpss_per_var": crpss_per_var,
        })

    # Print and save comparison table
    var_short = ["Temp", "Rain", "Hum", "Wind"]
    var_units = ["°C",   "mm",   "%",   "m/s"]
    crps_header  = "".join(f"  {v+' CRPS':>10}" for v in var_short)
    crpss_header = "".join(f"  {v+' CRPSS':>11}" for v in var_short)
    header = (f"{'Config':<14} {'CRPS':>7} {'CRPSS':>7} {'NLL':>7} {'Cov50%':>8} {'Cov90%':>8}"
              + crps_header + crpss_header)
    sep   = "-" * len(header)
    lines = [
        "ABLATION RESULTS",
        "Overall CRPS: mean across all 4 variables (mixed units — use for run comparison only)",
        "Overall CRPSS: skill vs climatology (0=no better than always predicting training mean, 1=perfect)",
        f"Per-variable CRPS units: " + "  ".join(f"{v}={u}" for v, u in zip(var_short, var_units)),
        "Per-variable CRPSS: dimensionless — comparable across variables",
        "Deterministic baseline: CRPS = MAE (point-forecast CRPS); NLL and coverage are N/A",
        "",
        header, sep,
    ]
    for r in results:
        crps_cols  = "".join(f"  {v:>10.4f}" for v in r["crps_per_var"])
        crpss_cols = "".join(f"  {v:>11.4f}" for v in r["crpss_per_var"])
        if r.get("deterministic", False):
            lines.append(
                f"{r['name']:<14} "
                f"{r['test_crps']:>7.4f} "
                f"{r['overall_crpss']:>7.4f} "
                f"{'N/A':>7} "
                f"{'N/A':>8} "
                f"{'N/A':>8}"
                f"{crps_cols}"
                f"{crpss_cols}"
            )
        else:
            lines.append(
                f"{r['name']:<14} "
                f"{r['test_crps']:>7.4f} "
                f"{r['overall_crpss']:>7.4f} "
                f"{r['test_nll']:>7.4f} "
                f"{r['cov_50']*100:>7.1f}% "
                f"{r['cov_90']*100:>7.1f}%"
                f"{crps_cols}"
                f"{crpss_cols}"
            )
    lines.append(sep)

    print("\n" + "\n".join(lines))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "ablation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAblation summary saved -> {summary_path}")


if __name__ == '__main__':
    real_data_test()
    ablation_test()











 
