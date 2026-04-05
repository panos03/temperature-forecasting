"""
task.py - Evaluation metrics for probabilistic multivariate weather forecasting

Provides standalone evaluation functions to be imported wherever needed.
Supports model outputs of shape (B, H, V) for means and std, where:
    B = batch size
    H = forecast horizon (7 days)
    V = number of variables (4: temperature, rainfall, humidity, wind speed)

"""

import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from train import (
    load_daily_data,
    ProbabilisticForecaster,
    WeatherWindowDataset,
    FEATURE_COLUMNS,
    TARGET_FEATURE_INDICES,
    PRECIP_TARGET_IDX,
    MODELS_DIR,
)


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

VAR_NAMES = [FEATURE_COLUMNS[i] for i in TARGET_FEATURE_INDICES]

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
# Shared: print metrics + save plots
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"

def evaluate_and_plot(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                      mu_plot: np.ndarray, sigma_plot: np.ndarray, y_plot: np.ndarray,
                      var_names: list, label: str):
    """
    Print all evaluation metrics and save calibration + forecast PNGs.

    Args:
        mu, sigma, y:             tensors in normalised space — used for metrics
        mu_plot, sigma_plot, y_plot: numpy arrays in real-world units — used for plots
        var_names:                list of variable name strings
        label:                    short tag used in output filenames, e.g. 'dummy' or 'real'
    """
    results = evaluate_all(mu, sigma, y, var_names=var_names)

    lines = []
    lines.append(f"=== Results ({label}) ===")
    lines.append(f"  NLL:  {results['nll']:.4f}")
    lines.append(f"  CRPS: {results['crps']:.4f}")
    lines.append(f"  RMSE: {results['rmse']:.4f}")
    lines.append(f"  MAE:  {results['mae']:.4f}")
    lines.append("  Coverage:")
    for cl, cov in results['coverage'].items():
        lines.append(f"    {int(cl*100):>3}% interval -> {cov*100:.1f}% empirical")
    lines.append("  RMSE per variable:")
    for var, val in results['rmse_per_var'].items():
        lines.append(f"    {var}: {val:.4f}")

    print()
    for line in lines:
        print(line)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n\n")
    print(f"  Summary appended -> {summary_path}")

    plot_calibration(mu, sigma, y,
                     save_path=os.path.join(RESULTS_DIR, f'calibration_{label}.png'))
    # Visualise the first test-set window as a single representative example
    plot_forecast(mu_plot[0], sigma_plot[0], y_plot[0],
                  var_names=var_names,
                  save_path=os.path.join(RESULTS_DIR, f'forecast_{label}.png'))


# ---------------------------------------------------------------------------
# Dummy-data smoke test — no real data required
# ---------------------------------------------------------------------------

def dummy_data_test():
    """Smoke test using randomly generated tensors."""
    B, H, V = 32, 7, 4
    mu    = torch.randn(B, H, V)
    sigma = torch.rand(B, H, V) + 0.1
    y     = torch.randn(B, H, V)

    evaluate_and_plot(mu, sigma, y,
                      mu.numpy(), sigma.numpy(), y.numpy(),
                      var_names=VAR_NAMES, label='dummy')
    print("\nDummy data test complete.")


# ---------------------------------------------------------------------------
# Real-data evaluation — loads model saved by train.py, no training here
# ---------------------------------------------------------------------------

def real_data_test():
    """
    Load the model saved by train.py, run inference on the test split, and
    produce forecast and calibration plots with real predicted values.

    Run train.py first (e.g. main(max_records=500, n_epochs=5)) to generate
    the required artefacts in the models/ directory.
    """
    weights_path = os.path.join(MODELS_DIR, "best_model.pt")
    stats_path   = os.path.join(MODELS_DIR, "normalisation_stats.pt")
    config_path  = os.path.join(MODELS_DIR, "split_config.pt")

    for p in (weights_path, stats_path, config_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found — run train.py first to generate model artefacts."
            )

    # ---- Load saved artefacts ----
    stats       = torch.load(stats_path,  map_location="cpu")
    config      = torch.load(config_path, map_location="cpu")
    train_mean  = stats["mean"]
    train_std   = stats["std"]
    max_records = config["max_records"]
    split_fracs = config["split_fracs"]

    # ---- Reconstruct test split using the same config (no retraining) ----
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

    # ---- Load model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ProbabilisticForecaster()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[EVAL] Loaded model from {weights_path}  (device: {device})")

    # ---- Collect predictions ----
    all_mu, all_sigma, all_p_rain, all_y = [], [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            mu, sigma, p_rain = model(x.to(device))
            all_mu.append(mu.cpu())
            all_sigma.append(sigma.cpu())
            all_p_rain.append(p_rain.cpu())
            all_y.append(y.cpu())

    mu_norm     = torch.cat(all_mu,     dim=0)
    sigma_norm  = torch.cat(all_sigma,  dim=0)
    p_rain_norm = torch.cat(all_p_rain, dim=0)
    y_norm      = torch.cat(all_y,      dim=0)

    # For precipitation, replace mu with the expected value: p_rain * mu_amount
    mu_norm = mu_norm.clone()
    mu_norm[..., PRECIP_TARGET_IDX] = (
        p_rain_norm.squeeze(-1) * mu_norm[..., PRECIP_TARGET_IDX]
    )

    # ---- Denormalise for interpretable plots ----
    t_mean     = train_mean[TARGET_FEATURE_INDICES]
    t_std      = train_std[TARGET_FEATURE_INDICES]
    mu_real    = (mu_norm    * t_std + t_mean).numpy()
    sigma_real = (sigma_norm * t_std).numpy()
    y_real     = (y_norm     * t_std + t_mean).numpy()

    evaluate_and_plot(mu_norm, sigma_norm, y_norm,
                      mu_real, sigma_real, y_real,
                      var_names=VAR_NAMES, label='real')
    print("\nReal data test complete.")


if __name__ == '__main__':
    #dummy_data_test()
    real_data_test()















 