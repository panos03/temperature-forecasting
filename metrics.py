"""
metrics.py - Evaluation metrics for probabilistic multivariate weather forecasting

Provides standalone evaluation functions to be imported wherever needed
(e.g. training loop for validation, or final test evaluation script).
Supports model outputs of shape (B, H, V) for means and std devs, where:
    B = batch size
    H = forecast horizon (7 days)
    V = number of variables (4: temperature, rainfall, humidity, wind speed)

Visualisations use Pillow only (no matplotlib) to comply with the comp0197-pt
submission environment constraints.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# 1. Gaussian Negative Log Likelihood (NLL)
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
# 2. CRPS — Continuous Ranked Probability Score
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


# ---------------------------------------------------------------------------
# 3. RMSE and MAE (on predicted means only)
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
# 4. Calibration — Empirical Coverage at Multiple Confidence Levels
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

    coverage = empirical_coverage(mu, sigma, y, confidence_levels)
    expected = list(coverage.keys())
    observed = list(coverage.values())

    # Canvas setup
    W, H_img = 500, 500
    pad = 60                          # padding for axes
    plot_w = W - 2 * pad
    plot_h = H_img - 2 * pad

    img = Image.new('RGB', (W, H_img), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    def to_px(ex, ob):
        """Map (expected, observed) in [0,1] to pixel coordinates."""
        px = pad + int(ex * plot_w)
        py = H_img - pad - int(ob * plot_h)
        return px, py

    # Draw axes
    draw.line([(pad, pad), (pad, H_img - pad)], fill=(0, 0, 0), width=2)          # y-axis
    draw.line([(pad, H_img - pad), (W - pad, H_img - pad)], fill=(0, 0, 0), width=2)  # x-axis

    # Diagonal: perfect calibration (dashed via short segments)
    n_dash = 20
    for i in range(n_dash):
        if i % 2 == 0:
            x0, y0 = to_px(i / n_dash, i / n_dash)
            x1, y1 = to_px((i + 1) / n_dash, (i + 1) / n_dash)
            draw.line([(x0, y0), (x1, y1)], fill=(160, 160, 160), width=1)

    # Model calibration line
    pts = [to_px(e, o) for e, o in zip(expected, observed)]
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=(70, 130, 180), width=2)

    # Dots at each point
    r = 5
    for px, py in pts:
        draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=(70, 130, 180))

    # Axis tick labels (every 0.2)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except OSError:
        font = ImageFont.load_default()

    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        # x-axis ticks
        tx, ty = to_px(v, 0)
        draw.text((tx - 8, ty + 5), f'{v:.1f}', fill=(0, 0, 0), font=font)
        # y-axis ticks
        tx2, ty2 = to_px(0, v)
        draw.text((tx2 - 38, ty2 - 6), f'{v:.1f}', fill=(0, 0, 0), font=font)

    # Axis labels and title
    draw.text((W // 2 - 55, H_img - 18), 'Expected coverage', fill=(0, 0, 0), font=font)
    draw.text((4, H_img // 2 - 50), 'Empirical', fill=(0, 0, 0), font=font)
    draw.text((4, H_img // 2 - 35), 'coverage', fill=(0, 0, 0), font=font)
    draw.text((pad, 10), 'Calibration - Reliability Diagram', fill=(0, 0, 0), font=font)

    img.save(save_path)
    print(f'Calibration plot saved to {save_path}')


def plot_forecast(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                  var_names: list, sample_idx: int = 0,
                  save_path: str = 'forecast.png'):
    """
    Saves predicted mean ± 1σ / ± 2σ bands against ground truth as a PNG. 
    One subplot row per forecast variable.
    """
    H_steps = mu.shape[1]
    V = len(var_names)

    panel_w, panel_h = 600, 160
    pad_l, pad_r, pad_t, pad_b = 70, 20, 30, 30
    plot_w = panel_w - pad_l - pad_r
    plot_h = panel_h - pad_t - pad_b

    total_h = panel_h * V + 30          # +30 for title bar
    img = Image.new('RGB', (panel_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
        font_sm = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    # Title
    draw.text((10, 6), '7-Day Probabilistic Forecast vs Ground Truth',
              fill=(0, 0, 0), font=font)

    for vi in range(V):
        m = mu[sample_idx, :, vi]
        s = sigma[sample_idx, :, vi]
        truth = y[sample_idx, :, vi]

        y_off = 30 + vi * panel_h       # vertical offset for this panel

        all_vals = np.concatenate([m - 2*s, m + 2*s, truth])
        v_min, v_max = all_vals.min(), all_vals.max()
        v_range = v_max - v_min if v_max != v_min else 1.0

        def to_px(day_idx, val):
            """Map day index and value to pixel coords within this panel."""
            px = pad_l + int(day_idx / (H_steps - 1) * plot_w)
            py = y_off + pad_t + int((1 - (val - v_min) / v_range) * plot_h)
            return px, py

        # Axes
        draw.line([(pad_l, y_off + pad_t),
                   (pad_l, y_off + pad_t + plot_h)], fill=(0, 0, 0), width=1)
        draw.line([(pad_l, y_off + pad_t + plot_h),
                   (pad_l + plot_w, y_off + pad_t + plot_h)], fill=(0, 0, 0), width=1)

        # ±2σ band (light blue fill via horizontal scanlines)
        for d in range(H_steps - 1):
            # Fill between consecutive day columns using a polygon
            p2s = [
                to_px(d,     m[d]     - 2*s[d]),
                to_px(d,     m[d]     + 2*s[d]),
                to_px(d + 1, m[d + 1] + 2*s[d + 1]),
                to_px(d + 1, m[d + 1] - 2*s[d + 1]),
            ]
            draw.polygon(p2s, fill=(173, 216, 230))   # light blue

        # ±1σ band (medium blue)
        for d in range(H_steps - 1):
            p1s = [
                to_px(d,     m[d]     - s[d]),
                to_px(d,     m[d]     + s[d]),
                to_px(d + 1, m[d + 1] + s[d + 1]),
                to_px(d + 1, m[d + 1] - s[d + 1]),
            ]
            draw.polygon(p1s, fill=(100, 149, 237))   # cornflower blue

        # Predicted mean line (dark blue)
        mean_pts = [to_px(d, m[d]) for d in range(H_steps)]
        draw.line(mean_pts, fill=(70, 130, 180), width=2)
        for px, py in mean_pts:
            draw.ellipse([(px-3, py-3), (px+3, py+3)], fill=(70, 130, 180))

        # Ground truth line (red)
        truth_pts = [to_px(d, truth[d]) for d in range(H_steps)]
        draw.line(truth_pts, fill=(205, 92, 92), width=2)
        for px, py in truth_pts:
            draw.ellipse([(px-3, py-3), (px+3, py+3)], fill=(205, 92, 92))

        # Y-axis label (variable name)
        draw.text((4, y_off + pad_t + plot_h // 2 - 6),
                  var_names[vi], fill=(0, 0, 0), font=font_sm)

        # Y-axis min/max tick labels
        draw.text((pad_l - 45, y_off + pad_t + plot_h - 8),
                  f'{v_min:.1f}', fill=(80, 80, 80), font=font_sm)
        draw.text((pad_l - 45, y_off + pad_t),
                  f'{v_max:.1f}', fill=(80, 80, 80), font=font_sm)

        # X-axis day labels (only on last panel)
        if vi == V - 1:
            for d in range(H_steps):
                px, _ = to_px(d, v_min)
                draw.text((px - 5, y_off + pad_t + plot_h + 4),
                          str(d + 1), fill=(80, 80, 80), font=font_sm)
            draw.text((pad_l + plot_w // 2 - 20, y_off + pad_t + plot_h + 16),
                      'Forecast day', fill=(0, 0, 0), font=font_sm)

    img.save(save_path)
    print(f'Forecast plot saved to {save_path}')


# ---------------------------------------------------------------------------
# 6. Summary Table — All Metrics at Once
# ---------------------------------------------------------------------------

VAR_NAMES = ['temperature', 'rainfall', 'humidity', 'wind_speed']

def evaluate_all(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                 var_names: list = None) -> dict:
    """
    Runs all metrics in one call. Convenient for test.py.
    """
    if var_names is None:
        var_names = VAR_NAMES

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
# Quick smoke-test — run this file directly to verify shapes / no crashes
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B, H, V = 32, 7, 4
    mu_dummy    = torch.randn(B, H, V)
    sigma_dummy = torch.rand(B, H, V) + 0.1   # ensure > 0
    y_dummy     = torch.randn(B, H, V)

    results = evaluate_all(mu_dummy, sigma_dummy, y_dummy)

    print("=== Smoke Test Results ===")
    print(f"  NLL:  {results['nll']:.4f}")
    print(f"  CRPS: {results['crps']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE:  {results['mae']:.4f}")
    print(f"  Coverage:")
    for cl, cov in results['coverage'].items():
        print(f"    {int(cl*100):>3}% interval -> {cov*100:.1f}% empirical")
    print(f"  RMSE per variable:")
    for var, val in results['rmse_per_var'].items():
        print(f"    {var}: {val:.4f}")

    # Calibration plot — saved as PNG via Pillow
    plot_calibration(mu_dummy, sigma_dummy, y_dummy, save_path='calibration_test.png')

    # Forecast plot — saved as PNG via Pillow
    plot_forecast(mu_dummy.numpy(), sigma_dummy.numpy(), y_dummy.numpy(),
                  var_names=VAR_NAMES, sample_idx=0, save_path='forecast_test.png')

    print("\nAll checks passed.")