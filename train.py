"""
train.py — Probabilistic Multivariate Weather Forecasting for Minas Gerais
============================================================================
Full pipeline: data retrieval -> preprocessing -> training -> evaluation.

Environment: comp0197-pt
    micromamba create --name comp0197-pt python=3.12 -y
    micromamba activate comp0197-pt
    pip install torch torchvision pillow --index-url https://download.pytorch.org/whl/cpu

Additional package (only needed if data not already downloaded):
    pip install cdsapi

Additional package if SSL certificate issues arise:
    pip install --upgrade certifi

Data: If data/raw/era5_minas_gerais_hourly.csv is already present, the download
step is skipped entirely and cdsapi is not required.
"""

import os
import csv
import math
import zipfile
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

# ERA5 download settings
LATITUDE = -21.5
LONGITUDE = -45.5
DATE_RANGE = "2004-01-01/2024-12-31"
ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]

# Paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
RAW_CSV = os.path.join(RAW_DIR, "era5_minas_gerais_hourly.csv")
DAILY_CSV = os.path.join(PROCESSED_DIR, "minas_gerais_daily_weather.csv")

# Data pipeline settings
INPUT_DAYS = 30       # Number of past days the model sees
TARGET_DAYS = 7       # Number of future days the model predicts
BATCH_SIZE = 64

# Chronological split years (inclusive)
TRAIN_END_YEAR = 2019     # Train: 2004-2019
VAL_END_YEAR = 2021       # Val:   2020-2021
                          # Test:  2022-2024

# Training settings
EPOCHS = 100
PRETRAIN_EPOCHS = 15    # MSE pretraining epochs before switching to CRPS
LEARNING_RATE = 3e-4
SEED = 42


def set_seed(seed=SEED):
    """
    Fix all random seeds and disable cuDNN non-determinism.
    Call once at the start of main() to ensure reproducible training runs.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Daily CSV column order (output of preprocessing)
DAILY_COLUMNS = [
    "date",
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "relative_humidity_mean_pct",
    "total_precipitation_mm",
    "wind_speed_mean_ms",
    "surface_pressure_mean_hpa",
    "sin_day",
    "cos_day",
]

# Feature columns (everything except date)
FEATURE_COLUMNS = DAILY_COLUMNS[1:]
NUM_FEATURES = len(FEATURE_COLUMNS)

# The 4 variables the model predicts probabilistically
TARGET_FEATURE_INDICES = [
    FEATURE_COLUMNS.index("temperature_mean_c"),
    FEATURE_COLUMNS.index("relative_humidity_mean_pct"),
    FEATURE_COLUMNS.index("total_precipitation_mm"),
    FEATURE_COLUMNS.index("wind_speed_mean_ms"),
]
NUM_TARGET_FEATURES = len(TARGET_FEATURE_INDICES)  # 4
# Index of precipitation within the 4 target variables (0=temp, 1=humidity, 2=precip, 3=wind)
PRECIP_TARGET_IDX   = TARGET_FEATURE_INDICES.index(FEATURE_COLUMNS.index("total_precipitation_mm"))
# Output head: (mu, log_sigma) per target variable per timestep
DECODER_OUTPUT_SIZE = NUM_TARGET_FEATURES * 2      # 8


# =============================================================================
# SECTION 1: DATA DOWNLOAD
# =============================================================================

def download_era5():
    """
    Download ERA5 timeseries data from the Copernicus Climate Data Store.
    Only runs if the raw CSV does not already exist.
    Requires: pip install cdsapi
    """
    if os.path.exists(RAW_CSV):
        print(f"[DOWNLOAD] SKIP - {RAW_CSV} already exists")
        return

    try:
        import cdsapi
    except ImportError:
        print("[DOWNLOAD] ERROR - cdsapi not installed and raw data not found.")
        print("  Either place era5_minas_gerais_hourly.csv in data/raw/")
        print("  or install cdsapi: pip install cdsapi")
        raise SystemExit(1)

    os.makedirs(RAW_DIR, exist_ok=True)

    client = cdsapi.Client()
    dataset = "reanalysis-era5-single-levels-timeseries"
    request = {
        "variable": ERA5_VARIABLES,
        "location": {"longitude": LONGITUDE, "latitude": LATITUDE},
        "date": [DATE_RANGE],
        "data_format": "csv",
    }

    download_path = os.path.join(RAW_DIR, "era5_download.zip")

    print(f"[DOWNLOAD] Requesting ERA5 data for {DATE_RANGE}...")
    client.retrieve(dataset, request).download(download_path)
    print(f"[DOWNLOAD] Downloaded to {download_path}")

    # CDS wraps output in a zip - extract the CSV
    print("[DOWNLOAD] Extracting CSV from zip...")
    with zipfile.ZipFile(download_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        if not csv_files:
            raise RuntimeError(f"No CSV in zip. Contents: {zf.namelist()}")
        zf.extract(csv_files[0], RAW_DIR)
        extracted = os.path.join(RAW_DIR, csv_files[0])
        os.rename(extracted, RAW_CSV)

    os.remove(download_path)
    print(f"[DOWNLOAD] Done - saved to {RAW_CSV}")


# =============================================================================
# SECTION 2: PREPROCESSING (hourly CSV -> daily CSV)
# =============================================================================

def compute_relative_humidity(temp_k, dewpoint_k):
    """
    Relative humidity (%) from temperature and dewpoint (both in Kelvin).

    Uses the Magnus formula:
        RH = 100 * exp(a*Td/(b+Td)) / exp(a*T/(b+T))
    where a=17.625, b=243.04 C, T and Td are in Celsius.

    When dewpoint equals temperature, RH = 100% (saturated air).
    The larger the gap, the drier the air.
    """
    temp_c = temp_k - 273.15
    dewpoint_c = dewpoint_k - 273.15

    a = 17.625
    b = 243.04

    rh = 100.0 * torch.exp((a * dewpoint_c) / (b + dewpoint_c)) / \
                  torch.exp((a * temp_c) / (b + temp_c))

    return torch.clamp(rh, 0.0, 100.0)


def compute_wind_speed(u, v):
    """
    Wind speed (m/s) from u (eastward) and v (northward) components.

    ERA5 provides wind as a 2D vector. Speed is the magnitude:
        speed = sqrt(u^2 + v^2)
    We discard direction - only magnitude matters for evapotranspiration.
    """
    return torch.sqrt(u ** 2 + v ** 2)


def preprocess_hourly_to_daily():
    """
    Read the hourly ERA5 CSV, compute derived variables, aggregate to
    daily values, add seasonal encoding, and save as a clean daily CSV.

    ERA5 CSV columns: valid_time, u10, v10, d2m, t2m, sp, tp, latitude, longitude

    Output columns:
        date, temperature_mean_c, temperature_min_c, temperature_max_c,
        relative_humidity_mean_pct, total_precipitation_mm,
        wind_speed_mean_ms, surface_pressure_mean_hpa, sin_day, cos_day
    """
    if os.path.exists(DAILY_CSV):
        print(f"[PREPROCESS] SKIP - {DAILY_CSV} already exists")
        return

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}. Run download first.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"[PREPROCESS] Loading {RAW_CSV}...")

    # --- Read hourly CSV ---
    with open(RAW_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[PREPROCESS] Loaded {len(rows)} hourly rows")

    # --- Parse into torch tensors ---
    times = [datetime.strptime(r["valid_time"], "%Y-%m-%d %H:%M:%S") for r in rows]
    dates_list = [t.date() for t in times]

    t2m = torch.tensor([float(r["t2m"]) for r in rows], dtype=torch.float64)
    d2m = torch.tensor([float(r["d2m"]) for r in rows], dtype=torch.float64)
    u10 = torch.tensor([float(r["u10"]) for r in rows], dtype=torch.float64)
    v10 = torch.tensor([float(r["v10"]) for r in rows], dtype=torch.float64)
    sp = torch.tensor([float(r["sp"]) for r in rows], dtype=torch.float64)
    tp = torch.tensor([float(r["tp"]) for r in rows], dtype=torch.float64)

    # --- Compute derived hourly variables ---
    temp_c = t2m - 273.15
    rh = compute_relative_humidity(t2m, d2m)
    wind_speed = compute_wind_speed(u10, v10)
    pressure_hpa = sp / 100.0
    precip_m = tp

    # --- Check for NaN values ---
    for name, arr in [("t2m", t2m), ("d2m", d2m), ("u10", u10),
                      ("v10", v10), ("sp", sp), ("tp", tp)]:
        nan_count = torch.isnan(arr).sum().item()
        if nan_count > 0:
            print(f"[PREPROCESS] WARNING: {nan_count} NaN values in {name}")

    # --- Group by date and compute daily aggregates ---
    print("[PREPROCESS] Aggregating to daily values...")

    day_indices = defaultdict(list)
    for i, d in enumerate(dates_list):
        day_indices[d].append(i)

    daily_rows = []
    for d in sorted(day_indices.keys()):
        idx = day_indices[d]

        day_of_year = d.timetuple().tm_yday

        daily_rows.append({
            "date": d.isoformat(),
            "temperature_mean_c": round(torch.mean(temp_c[idx]).item(), 4),
            "temperature_min_c": round(torch.min(temp_c[idx]).item(), 4),
            "temperature_max_c": round(torch.max(temp_c[idx]).item(), 4),
            "relative_humidity_mean_pct": round(torch.mean(rh[idx]).item(), 4),
            "total_precipitation_mm": round(torch.sum(precip_m[idx]).item() * 1000, 4),
            "wind_speed_mean_ms": round(torch.mean(wind_speed[idx]).item(), 4),
            "surface_pressure_mean_hpa": round(torch.mean(pressure_hpa[idx]).item(), 4),
            "sin_day": round(math.sin(2 * math.pi * day_of_year / 365.25), 6),
            "cos_day": round(math.cos(2 * math.pi * day_of_year / 365.25), 6),
        })

    # --- Save daily CSV ---
    with open(DAILY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DAILY_COLUMNS)
        writer.writeheader()
        writer.writerows(daily_rows)

    print(f"[PREPROCESS] Done - saved {len(daily_rows)} daily rows to {DAILY_CSV}")
    print(f"[PREPROCESS] Date range: {daily_rows[0]['date']} to {daily_rows[-1]['date']}")


# =============================================================================
# SECTION 3: LOAD, NORMALISE, AND SPLIT
# =============================================================================

def load_daily_data(max_records=None):
    """
    Load the daily CSV into a torch tensor and date list.

    Args:
        max_records: choose to take only first N rows - useful for quick
                     tests without loading the full 20-year dataset.

    Returns:
        dates:    list of date strings ["2004-01-01", ...]
        features: torch.Tensor of shape (num_days, NUM_FEATURES)
    """
    if not os.path.exists(DAILY_CSV):
        raise FileNotFoundError(f"Daily CSV not found: {DAILY_CSV}. Run preprocessing first.")

    with open(DAILY_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_records is not None:
        rows = rows[:max_records]

    dates = [r["date"] for r in rows]
    features = torch.tensor(
        [[float(r[col]) for col in FEATURE_COLUMNS] for r in rows],
        dtype=torch.float32,
    )

    print(f"[DATA] Loaded {len(dates)} days, {NUM_FEATURES} features")
    return dates, features


def split_and_normalise(dates, features, split_fracs=None):
    """
    Chronological train/val/test split with z-score normalisation.

    Normalisation is fit on the TRAINING set only to prevent data leakage.
    The same mean and std are applied to val and test sets.

    Args:
        dates:       list of date strings
        features:    torch.Tensor (num_days, NUM_FEATURES)
        split_fracs: optional tuple (train_frac, val_frac, test_frac) that
                     overrides the year-based split with a simple index split.
                     Use this when max_records is set and the slice doesn't
                     span multiple calendar years

    Returns:
        train_data, val_data, test_data: torch.Tensors (normalised)
        train_mean, train_std: torch.Tensors for inverse transform later
    """
    if split_fracs is not None:
        n = len(features)
        n_train = int(n * split_fracs[0])
        n_val   = int(n * split_fracs[1])
        train_idx = list(range(0, n_train))
        val_idx   = list(range(n_train, n_train + n_val))
        test_idx  = list(range(n_train + n_val, n))
    else:
        # Default: year-based chronological split
        train_idx = [i for i, d in enumerate(dates) if int(d[:4]) <= TRAIN_END_YEAR]
        val_idx   = [i for i, d in enumerate(dates) if TRAIN_END_YEAR < int(d[:4]) <= VAL_END_YEAR]
        test_idx  = [i for i, d in enumerate(dates) if int(d[:4]) > VAL_END_YEAR]

    train_data = features[train_idx]
    val_data   = features[val_idx]
    test_data  = features[test_idx]

    print(f"[SPLIT] Train: {len(train_idx)} days ({dates[train_idx[0]]} to {dates[train_idx[-1]]})")
    print(f"[SPLIT] Val:   {len(val_idx)} days ({dates[val_idx[0]]} to {dates[val_idx[-1]]})")
    print(f"[SPLIT] Test:  {len(test_idx)} days ({dates[test_idx[0]]} to {dates[test_idx[-1]]})")

    # Z-score normalisation - fit on train only
    train_mean = train_data.mean(dim=0)
    train_std = train_data.std(dim=0)

    # Prevent division by zero for any constant feature
    train_std[train_std == 0] = 1.0

    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    print(f"[NORMALISE] Mean: {train_mean.round(decimals=2).tolist()}")
    print(f"[NORMALISE] Std:  {train_std.round(decimals=2).tolist()}")

    return train_data, val_data, test_data, train_mean, train_std


# =============================================================================
# SECTION 4: PYTORCH DATASET AND DATALOADER
# =============================================================================

class WeatherWindowDataset(Dataset):
    """
    Sliding window dataset for time series forecasting.

    Given a sequence of daily weather observations, creates overlapping
    (input_window, target_window) pairs:
        input:  days [i, i+INPUT_DAYS)         -> shape (INPUT_DAYS, NUM_FEATURES)
        target: days [i+INPUT_DAYS, i+INPUT_DAYS+TARGET_DAYS) -> shape (TARGET_DAYS, NUM_FEATURES)
    """

    def __init__(self, data, input_days=INPUT_DAYS, target_days=TARGET_DAYS):
        """
        Args:
            data: torch.Tensor of shape (num_days, num_features), normalised
            input_days: number of past days in each input window
            target_days: number of future days to predict
        """
        self.data = data
        self.input_days = input_days
        self.target_days = target_days
        self.total_window = input_days + target_days
        self.num_samples = len(data) - self.total_window + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"Data too short ({len(data)} days) for window size "
                f"({input_days} + {target_days} = {self.total_window})"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_days]                                      # (INPUT_DAYS, NUM_FEATURES)
        y_full = self.data[idx + self.input_days: idx + self.total_window]             # (TARGET_DAYS, NUM_FEATURES)
        # Extract only the 4 target variables from the full 9-feature target
        y = y_full[:, TARGET_FEATURE_INDICES]                                          # (TARGET_DAYS, NUM_TARGET_FEATURES)
        return x, y


def create_dataloaders(train_data, val_data, test_data, batch_size=BATCH_SIZE):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    Training set is shuffled (shuffling windows across epochs is fine -
    each window maintains its internal temporal order).
    Val and test are not shuffled.
    """
    train_dataset = WeatherWindowDataset(train_data)
    val_dataset = WeatherWindowDataset(val_data)
    test_dataset = WeatherWindowDataset(test_data)

    print(f"[DATALOADER] Train windows: {len(train_dataset)}")
    print(f"[DATALOADER] Val windows:   {len(val_dataset)}")
    print(f"[DATALOADER] Test windows:  {len(test_dataset)}")

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =============================================================================
# SECTION 5: MODEL
# =============================================================================

class Encoder(torch.nn.Module):
    """
    Multi-layer LSTM encoder.

    Reads the full 30-day input sequence (all 9 features) and compresses
    it into a fixed-size context vector via the final hidden state.

    Args:
        num_features:  Number of input features (9)
        hidden_size:   LSTM hidden dimension
        num_layers:    Number of stacked LSTM layers
        dropout:       Dropout between LSTM layers (0 if num_layers == 1)
    """
    def __init__(self, num_features, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        """
        Args:
            x: (batch, INPUT_DAYS, num_features)
        Returns:
            h: (num_layers, batch, hidden_size) — hidden state for decoder init
            c: (num_layers, batch, hidden_size) — cell state for decoder init
        """
        _, (h, c) = self.lstm(x)
        return h, c


class Decoder(torch.nn.Module):
    """
    Autoregressive LSTM decoder with a zero-inflated precipitation head.

    At each step the LSTM input is the previous step's predicted mu values
    (or ground truth during training — teacher forcing).
    Step 0 is seeded with a learned start embedding.

    The standard output_head produces (mu, log_sigma) for all 4 target
    variables. A separate precip_head produces logit(p_rain), the probability
    of non-zero precipitation, used by the zero-inflated loss.

    Args:
        hidden_size:        Must match encoder hidden_size
        num_layers:         Must match encoder num_layers
        target_days:        Forecast horizon (7)
        num_targets:        Number of predicted variables (4)
        num_input_features: Feature size for teacher forcing input (4)
        dropout:            Dropout between LSTM layers
    """
    def __init__(self, hidden_size, num_layers, target_days, num_targets,
                 num_input_features=NUM_TARGET_FEATURES, dropout=0.2):
        super().__init__()
        self.target_days = target_days
        self.num_targets = num_targets

        self.start_embedding = torch.nn.Parameter(
            torch.zeros(1, 1, num_input_features)
        )
        self.input_proj  = torch.nn.Linear(num_input_features, hidden_size)
        self.lstm        = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = torch.nn.Linear(hidden_size, num_targets * 2)
        # Separate head for precipitation zero-inflation probability
        self.precip_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, h, c, targets=None, teacher_forcing_ratio=0.5):
        """
        Args:
            h:                    (num_layers, batch, hidden_size) from encoder
            c:                    (num_layers, batch, hidden_size) from encoder
            targets:              (batch, target_days, num_targets) or None
            teacher_forcing_ratio: P(use ground truth as next input)

        Returns:
            mu:     (batch, target_days, num_targets)
            sigma:  (batch, target_days, num_targets)
            p_rain: (batch, target_days, 1) — P(precipitation > 0)
        """
        batch_size = h.shape[1]
        all_mu, all_sigma, all_p_rain = [], [], []

        current_input = self.start_embedding.expand(batch_size, 1, -1)

        for step in range(self.target_days):
            step_input       = self.input_proj(current_input)    # (batch, 1, hidden_size)
            out, (h, c)      = self.lstm(step_input, (h, c))    # (batch, 1, hidden_size)
            out_flat         = out.squeeze(1)                    # (batch, hidden_size)

            projected = self.output_head(out_flat)               # (batch, num_targets*2)
            projected = projected.view(batch_size, self.num_targets, 2)

            mu        = projected[..., 0]                        # (batch, num_targets)
            log_sigma = torch.clamp(projected[..., 1], min=-6.0, max=2.0)
            sigma     = torch.exp(log_sigma)
            p_rain    = torch.sigmoid(self.precip_head(out_flat))  # (batch, 1)

            all_mu.append(mu)
            all_sigma.append(sigma)
            all_p_rain.append(p_rain)

            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_input = targets[:, step, :]
            else:
                next_input = mu.detach()
            current_input = next_input.unsqueeze(1)

        mu_all     = torch.stack(all_mu,     dim=1)   # (batch, target_days, num_targets)
        sigma_all  = torch.stack(all_sigma,  dim=1)
        p_rain_all = torch.stack(all_p_rain, dim=1)   # (batch, target_days, 1)

        return mu_all, sigma_all, p_rain_all


class ProbabilisticForecaster(torch.nn.Module):
    """
    Full encoder-decoder model for probabilistic multi-step weather forecasting.

    Architecture summary:
        Encoder LSTM  (9 features in -> hidden_size)
              |
           (h, c)
              |
        Decoder LSTM  (learned start embedding -> hidden_size)
              |
        output_head   (hidden_size -> 4*2)   mu, sigma for all variables
        precip_head   (hidden_size -> 1)     logit(p_rain) for precipitation
              |
        mu, sigma, p_rain   (batch, 7, 4) / (batch, 7, 4) / (batch, 7, 1)

    Designed for a T4 GPU on Google Colab:
        hidden_size=128, num_layers=2 -> ~470k parameters, fits comfortably
        in 16 GB VRAM with batch_size=64 and sequence length 30+7.

    Args:
        num_features:  Total encoder input features (9)
        hidden_size:   LSTM hidden dimension (default 128)
        num_layers:    Stacked LSTM layers in both encoder and decoder (default 2)
        target_days:   Forecast horizon (default 7)
        num_targets:   Variables to predict probabilistically (default 4)
        dropout:       Regularisation dropout between LSTM layers (default 0.2)
    """
    def __init__(
        self,
        num_features=NUM_FEATURES,
        hidden_size=128,
        num_layers=2,
        target_days=TARGET_DAYS,
        num_targets=NUM_TARGET_FEATURES,
        dropout=0.3,
    ):
        super().__init__()
        self.encoder = Encoder(num_features, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, num_layers, target_days, num_targets, dropout=dropout)

    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        """
        Args:
            x:                    (batch, INPUT_DAYS, NUM_FEATURES)
            targets:              (batch, TARGET_DAYS, NUM_TARGET_FEATURES) or None
            teacher_forcing_ratio: passed to decoder (ignored at inference)
        """
        h, c = self.encoder(x)
        mu, sigma, p_rain = self.decoder(h, c, targets=targets,
                                         teacher_forcing_ratio=teacher_forcing_ratio)
        return mu, sigma, p_rain

# =============================================================================
# SECTION 6: LOSS FUNCTION AND TRAINING LOOP
# =============================================================================

def mse_loss_mu(mu, target):
    """MSE on predicted means only — used for phase-1 pretraining."""
    return torch.mean((mu - target) ** 2)


def beta_nll_loss(mu, sigma, target, beta=0.5):
    """
    Beta-NLL loss (Seitzer et al. 2022).

    Weights each NLL term by sigma^(2*beta) with a stop-gradient on sigma.
    This breaks the feedback loop where the model lowers sigma to reduce loss
    without improving mu — the core cause of sigma collapse.

    beta=0 reduces to plain NLL; beta=1 weights fully by variance.
    beta=0.5 is the recommended default.

    Args:
        mu:     (batch, TARGET_DAYS, NUM_TARGET_FEATURES)
        sigma:  (batch, TARGET_DAYS, NUM_TARGET_FEATURES)
        target: (batch, TARGET_DAYS, NUM_TARGET_FEATURES)
        beta:   weighting exponent in [0, 1]
    """
    variance = sigma ** 2
    weight   = variance.detach() ** beta
    return 0.5 * (weight * (torch.log(variance) + (target - mu) ** 2 / variance)).mean()


def crps_gaussian_loss(mu, sigma, y):
    """
    Closed-form CRPS for Gaussian predictive distributions.

    Used as the validation metric for early stopping. CRPS is expressed in
    the same units as the forecast variables, is robust to sigma scale, and
    jointly rewards both accuracy and calibration.

    Lower is better.
    """
    sigma  = torch.clamp(sigma, min=1e-6)
    z      = (y - mu) / sigma
    normal = torch.distributions.Normal(0, 1)
    phi    = torch.exp(normal.log_prob(z))
    Phi    = normal.cdf(z)
    crps   = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / math.sqrt(math.pi))
    return crps.mean()


def zero_inflated_crps_loss(mu, sigma, p_rain, y):
    """
    CRPS loss with a zero-inflated auxiliary BCE term for precipitation.

    All 4 variables receive full Gaussian CRPS — sigma always gets gradient
    on every day for every variable. A separate BCE term trains p_rain to
    predict rain occurrence independently, without scaling or blocking the
    CRPS gradient.

    The previous formulation (p * crps_p) starved sigma of gradient on dry
    days (~40% of data), causing systematic overconfidence across all variables.

    Args:
        mu:     (batch, TARGET_DAYS, NUM_TARGET_FEATURES)
        sigma:  (batch, TARGET_DAYS, NUM_TARGET_FEATURES)
        p_rain: (batch, TARGET_DAYS, 1) — P(precipitation > 0)
        y:      (batch, TARGET_DAYS, NUM_TARGET_FEATURES)
    """
    # Full Gaussian CRPS for all 4 variables — unweighted
    crps_all = crps_gaussian_loss(mu, sigma, y)

    # Auxiliary BCE to train p_rain for rain/no-rain classification
    p      = p_rain.squeeze(-1).clamp(1e-6, 1 - 1e-6)
    y_rain = (y[..., PRECIP_TARGET_IDX] > 0).float()
    bce    = -(y_rain * torch.log(p) + (1 - y_rain) * torch.log(1 - p)).mean()

    return crps_all + 0.5 * bce


def train_one_epoch(model, loader, optimiser, device, teacher_forcing_ratio=0.5, phase=2):
    """phase=1: MSE on mu only  |  phase=2: zero-inflated CRPS"""
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimiser.zero_grad()
        mu, sigma, p_rain = model(x, targets=y, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = mse_loss_mu(mu, y) if phase == 1 else zero_inflated_crps_loss(mu, sigma, p_rain, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, metric='crps'):
    """
    Evaluate the model on a validation or test DataLoader.

    Args:
        model:   ProbabilisticForecaster
        loader:  DataLoader (val or test)
        device:  torch.device
        metric:  'crps' (default) or 'nll' — which loss to return

    Returns:
        Mean loss over all batches.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            mu, sigma, p_rain = model(x)
            if metric == 'crps':
                total_loss += zero_inflated_crps_loss(mu, sigma, p_rain, y).item()
            else:
                total_loss += beta_nll_loss(mu, sigma, y).item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader, device, weights_path,
          n_epochs=EPOCHS, pretrain_epochs=PRETRAIN_EPOCHS):
    """
    Two-phase training loop with CRPS-based early stopping.

    Phase 1 — MSE pretraining (pretrain_epochs, no early stopping, TF=1.0):
        Trains mu to convergence before sigma is introduced. Prevents the
        zero-inflated precipitation BCE component from dominating early
        training and starving sigma of gradient.

    Phase 2 — CRPS training (up to n_epochs, early stopping on val CRPS):
        Trains on zero_inflated_crps_loss. Teacher forcing decays linearly
        1.0 → 0.0. Best checkpoint selected by validation CRPS.

    Args:
        model:           ProbabilisticForecaster
        train_loader:    training DataLoader
        val_loader:      validation DataLoader
        device:          torch.device
        weights_path:    path to save best model .pt file
        n_epochs:        max phase-2 epochs (default: EPOCHS)
        pretrain_epochs: phase-1 MSE epochs (default: PRETRAIN_EPOCHS)

    Returns:
        history: dict with keys "phase1_train_mse", "train_loss", "val_crps"
    """
    PATIENCE = 15

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )

    history = {"phase1_train_mse": [], "train_loss": [], "val_crps": []}

    # -------------------------------------------------------------------------
    # Phase 1: MSE pretraining
    # -------------------------------------------------------------------------
    print(f"\n[TRAIN] Phase 1: MSE pretraining for {pretrain_epochs} epochs")
    print(f"[TRAIN] Device: {device}\n")

    for epoch in range(1, pretrain_epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimiser, device,
                                    teacher_forcing_ratio=1.0, phase=1)
        history["phase1_train_mse"].append(train_mse)
        print(f"  Phase1 Epoch {epoch:02d}/{pretrain_epochs} | Train MSE: {train_mse:.4f}")

    # -------------------------------------------------------------------------
    # Phase 2: CRPS training with early stopping
    # -------------------------------------------------------------------------
    best_val_crps = float("inf")
    epochs_without_improvement = 0

    print(f"\n[TRAIN] Phase 2: CRPS training for up to {n_epochs} epochs "
          f"(early stopping patience={PATIENCE})")
    print(f"[TRAIN] Best weights will be saved to: {weights_path}\n")

    for epoch in range(1, n_epochs + 1):
        tf_ratio   = max(0.0, 1.0 - (epoch - 1) / n_epochs)
        train_loss = train_one_epoch(model, train_loader, optimiser, device,
                                     teacher_forcing_ratio=tf_ratio, phase=2)
        val_crps   = validate(model, val_loader, device, metric='crps')

        scheduler.step(val_crps)

        history["train_loss"].append(train_loss)
        history["val_crps"].append(val_crps)

        improved = val_crps < best_val_crps
        if improved:
            best_val_crps = val_crps
            epochs_without_improvement = 0
            torch.save(model.state_dict(), weights_path)
            marker = " *"
        else:
            epochs_without_improvement += 1
            marker = ""

        print(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"Train CRPS: {train_loss:.4f} | "
            f"Val CRPS: {val_crps:.4f} | "
            f"TF: {tf_ratio:.2f}"
            f"{marker}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"\n[TRAIN] Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    print(f"\n[TRAIN] Done. Best val CRPS: {best_val_crps:.4f} — weights saved to {weights_path}")
    return history


# =============================================================================
# SECTION 7: EVALUATION
# =============================================================================

# TODO: Implement evaluation metrics here.
#
# Metrics to compute on the test set:
#   - CRPS (Continuous Ranked Probability Score): standard for probabilistic forecasts
#   - Calibration: does the model's 90% interval contain the truth ~90% of the time?
#   - Sharpness: how tight are the predicted intervals?
#   - MSE/MAE of the mean predictions (for comparison with baselines)
#
# Baselines to compare against:
#   - Persistence: predict tomorrow = today
#   - Climatological average: predict the historical mean for that day of year


# =============================================================================
# SECTION 8: MAIN
# =============================================================================

def main(max_records=None, n_epochs=EPOCHS):
    """
    Full training pipeline.

    Args:
        max_records: if set, use only the first N days — useful for a quick
                     smoke test before running on the full dataset.
                     Switches the train/val/test split to a 70/15/15 fraction
                     split instead of the calendar-year split.
        n_epochs:    maximum training epochs (default: EPOCHS constant)
    """
    set_seed()
    print("=" * 70)
    print("Probabilistic Weather Forecasting - Minas Gerais Coffee Belt")
    print("=" * 70)

    # --- Step 1: Ensure raw data exists ---
    download_era5()

    # --- Step 2: Preprocess hourly -> daily ---
    preprocess_hourly_to_daily()

    # --- Step 3: Load, split, normalise ---
    dates, features = load_daily_data(max_records=max_records)
    # Use fraction-based split when a record limit is set — the slice won't
    # span the 2019/2021 year boundaries used by the default calendar split.
    split_fracs = (0.70, 0.15, 0.15) if max_records is not None else None
    train_data, val_data, test_data, train_mean, train_std = split_and_normalise(
        dates, features, split_fracs=split_fracs
    )

    # --- Step 4: Create DataLoaders ---
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

    # Verify shapes
    x_batch, y_batch = next(iter(train_loader))
    print(f"\n[VERIFY] Input batch shape:  {x_batch.shape}")   # (batch, 30, 9)
    print(f"[VERIFY] Target batch shape: {y_batch.shape}")      # (batch, 7, 4)

    # --- Step 5: Create model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProbabilisticForecaster()
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] ProbabilisticForecaster — {num_params:,} trainable parameters")
    print(f"[MODEL] Device: {device}")

    # --- Step 6: Train ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    weights_path = os.path.join(MODELS_DIR, "best_model.pt")
    # NOTE: weights are saved every epoch when validation improves, inside train()
    history = train(model, train_loader, val_loader, device, weights_path,
                    n_epochs=n_epochs)
    history_path = os.path.join(MODELS_DIR, "loss_history.pt")
    torch.save(history, history_path)
    print(f"[TRAIN] Loss history saved to {history_path}")

    # --- Step 7: Evaluate on test set ---
    print("\n[EVAL] Loading best weights for test evaluation...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    test_crps = validate(model, test_loader, device, metric='crps')
    test_nll  = validate(model, test_loader, device, metric='nll')
    print(f"[EVAL] Test CRPS: {test_crps:.4f}")
    print(f"[EVAL] Test Beta-NLL: {test_nll:.4f}")

    # --- Step 8: Save normalisation stats and split config ---
    stats_path = os.path.join(MODELS_DIR, "normalisation_stats.pt")
    torch.save({"mean": train_mean, "std": train_std}, stats_path)
    print(f"[EVAL] Normalisation stats saved to {stats_path}")

    # split_config lets task.py reconstruct the exact same test split
    config_path = os.path.join(MODELS_DIR, "split_config.pt")
    torch.save({"max_records": max_records, "split_fracs": split_fracs}, config_path)
    print(f"[EVAL] Split config saved to {config_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    # For a quick subset test use: main(max_records=500, n_epochs=20)
    main()
