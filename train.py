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
WEIGHTS_DIR = "weights"
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
LEARNING_RATE = 1e-3

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

def load_daily_data():
    """
    Load the daily CSV into a torch tensor and date list.
    Returns:
        dates:    list of date strings ["2004-01-01", ...]
        features: torch.Tensor of shape (num_days, NUM_FEATURES)
    """
    if not os.path.exists(DAILY_CSV):
        raise FileNotFoundError(f"Daily CSV not found: {DAILY_CSV}. Run preprocessing first.")

    with open(DAILY_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    dates = [r["date"] for r in rows]
    features = torch.tensor(
        [[float(r[col]) for col in FEATURE_COLUMNS] for r in rows],
        dtype=torch.float32,
    )

    print(f"[DATA] Loaded {len(dates)} days, {NUM_FEATURES} features")
    return dates, features


def split_and_normalise(dates, features):
    """
    Chronological train/val/test split with z-score normalisation.

    Normalisation is fit on the TRAINING set only to prevent data leakage.
    The same mean and std are applied to val and test sets.

    Returns:
        train_data, val_data, test_data: torch.Tensors (normalised)
        train_mean, train_std: torch.Tensors for inverse transform later
    """
    # Find split indices based on year
    train_idx = [i for i, d in enumerate(dates) if int(d[:4]) <= TRAIN_END_YEAR]
    val_idx = [i for i, d in enumerate(dates) if TRAIN_END_YEAR < int(d[:4]) <= VAL_END_YEAR]
    test_idx = [i for i, d in enumerate(dates) if int(d[:4]) > VAL_END_YEAR]

    train_data = features[train_idx]
    val_data = features[val_idx]
    test_data = features[test_idx]

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
        x = self.data[idx: idx + self.input_days]                             # (INPUT_DAYS, NUM_FEATURES)
        y = self.data[idx + self.input_days: idx + self.total_window]          # (TARGET_DAYS, NUM_FEATURES)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =============================================================================
# SECTION 5: MODEL
# =============================================================================

# TODO: Define your probabilistic model here.
#
# The model should:
#   - Take input of shape (batch, INPUT_DAYS, NUM_FEATURES)
#   - Output distribution parameters of shape (batch, TARGET_DAYS, NUM_FEATURES, 2)
#     where the last dimension is (mu, sigma) for a Gaussian output
#   - Use an LSTM or Transformer encoder + a probabilistic prediction head
#
# Example skeleton:
#
# class ProbabilisticForecaster(torch.nn.Module):
#     def __init__(self, num_features, hidden_size, num_layers, target_days):
#         super().__init__()
#         self.encoder = torch.nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)
#         self.mu_head = torch.nn.Linear(hidden_size, target_days * num_features)
#         self.sigma_head = torch.nn.Linear(hidden_size, target_days * num_features)
#         self.target_days = target_days
#         self.num_features = num_features
#
#     def forward(self, x):
#         _, (h, _) = self.encoder(x)
#         h = h[-1]  # last layer hidden state
#         mu = self.mu_head(h).view(-1, self.target_days, self.num_features)
#         sigma = torch.nn.functional.softplus(self.sigma_head(h)).view(-1, self.target_days, self.num_features)
#         return mu, sigma


# =============================================================================
# SECTION 6: TRAINING LOOP
# =============================================================================

# TODO: Implement training loop here.
#
# Key points:
#   - Loss: Negative log-likelihood (NLL) of the true values under the
#     predicted Gaussian distribution:
#       NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
#   - Optimiser: Adam with learning rate 1e-3
#   - Track training and validation loss per epoch
#   - Save best model weights based on validation loss
#
# def gaussian_nll_loss(mu, sigma, target):
#     """Negative log-likelihood for a Gaussian distribution."""
#     variance = sigma ** 2
#     return 0.5 * torch.mean(torch.log(variance) + (target - mu) ** 2 / variance)
#
# def train_one_epoch(model, loader, optimiser):
#     ...
#
# def validate(model, loader):
#     ...


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

def main():
    print("=" * 70)
    print("Probabilistic Weather Forecasting - Minas Gerais Coffee Belt")
    print("=" * 70)

    # --- Step 1: Ensure raw data exists ---
    download_era5()

    # --- Step 2: Preprocess hourly -> daily ---
    preprocess_hourly_to_daily()

    # --- Step 3: Load, split, normalise ---
    dates, features = load_daily_data()
    train_data, val_data, test_data, train_mean, train_std = split_and_normalise(dates, features)

    # --- Step 4: Create DataLoaders ---
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

    # Verify shapes
    x_batch, y_batch = next(iter(train_loader))
    print(f"\n[VERIFY] Input batch shape:  {x_batch.shape}")   # (batch, 30, 9)
    print(f"[VERIFY] Target batch shape: {y_batch.shape}")      # (batch, 7, 9)

    # --- Step 5: Create model ---
    # TODO: model = ProbabilisticForecaster(...)

    # --- Step 6: Train ---
    # TODO: training loop

    # --- Step 7: Evaluate ---
    # TODO: evaluation on test set

    # --- Step 8: Save weights ---
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    # TODO: torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_model.pt"))

    print("\n[DONE]")


if __name__ == "__main__":
    main()