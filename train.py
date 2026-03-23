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

TARGET_FEATURE_INDICES = [
    FEATURE_COLUMNS.index("temperature_mean_c"),
    FEATURE_COLUMNS.index("relative_humidity_mean_pct"),
    FEATURE_COLUMNS.index("total_precipitation_mm"),
    FEATURE_COLUMNS.index("wind_speed_mean_ms"),
]
NUM_TARGET_FEATURES = len(TARGET_FEATURE_INDICES)  # 4
# Output head: (mu, log_sigma) per target variable per timestep
DECODER_OUTPUT_SIZE = NUM_TARGET_FEATURES * 2      # 8


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
    Multi-layer LSTM decoder with learned step embeddings.

    Rather than feeding predictions back in autoregressively (which can cause
    compounding errors during training), each decoder step is seeded with a
    learned positional embedding — one per forecast horizon step (7 total).
    The encoder hidden/cell states are used to initialise the decoder LSTM,
    so the context from the input window is fully preserved.

    At each step, a linear head maps the LSTM output to:
        (mu_1, log_sigma_1, mu_2, log_sigma_2, ...) for the 4 target variables
    sigma is recovered as exp(log_sigma), which is always positive and avoids
    the saturation issues of softplus near zero.

    Args:
        hidden_size:   Must match encoder hidden_size
        num_layers:    Must match encoder num_layers
        target_days:   Forecast horizon (7)
        num_targets:   Number of predicted variables (4)
        dropout:       Dropout between LSTM layers
    """
    def __init__(self, hidden_size, num_layers, target_days, num_targets, dropout=0.2):
        super().__init__()
        self.target_days = target_days
        self.num_targets = num_targets

        # One learned vector per forecast step — replaces autoregressive input
        # Shape: (target_days, hidden_size) so each step gets a full-width seed
        self.step_embeddings = torch.nn.Embedding(target_days, hidden_size)

        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Single linear head: hidden -> (mu, log_sigma) for each target variable
        self.output_head = torch.nn.Linear(hidden_size, num_targets * 2)

    def forward(self, h, c):
        """
        Args:
            h: (num_layers, batch, hidden_size) from encoder
            c: (num_layers, batch, hidden_size) from encoder
        Returns:
            mu:    (batch, target_days, num_targets)
            sigma: (batch, target_days, num_targets) — always positive via exp
        """
        batch_size = h.shape[1]

        # Build step embedding sequence: (batch, target_days, hidden_size)
        step_ids = torch.arange(self.target_days, device=h.device)          # (target_days,)
        step_emb = self.step_embeddings(step_ids)                            # (target_days, hidden_size)
        step_emb = step_emb.unsqueeze(0).expand(batch_size, -1, -1)         # (batch, target_days, hidden_size)

        # Run decoder LSTM over all 7 steps in one pass (teacher-forcing equivalent)
        out, _ = self.lstm(step_emb, (h, c))                                 # (batch, target_days, hidden_size)

        # Project each step's hidden state to (mu, log_sigma) pairs
        projected = self.output_head(out)                                    # (batch, target_days, num_targets*2)
        projected = projected.view(batch_size, self.target_days, self.num_targets, 2)

        mu = projected[..., 0]                                               # (batch, target_days, num_targets)
        log_sigma = projected[..., 1]
        # Clamp log_sigma for numerical stability — prevents sigma collapsing
        # to ~0 or exploding to inf in early training
        log_sigma = torch.clamp(log_sigma, min=-6.0, max=2.0)
        sigma = torch.exp(log_sigma)                                         # always positive

        return mu, sigma


class ProbabilisticForecaster(torch.nn.Module):
    """
    Full encoder-decoder model for probabilistic multi-step weather forecasting.

    Architecture summary:
        Encoder LSTM  (9 features in  -> hidden_size)
              |
        hidden state h, cell state c
              |
        Decoder LSTM  (learned step embeddings -> hidden_size)
              |
        Linear head   (hidden_size -> 4 * 2)
              |
        mu, sigma     (batch, 7, 4)

    Designed for a T4 GPU on Google Colab:
        hidden_size=256, num_layers=2 -> ~2.5M parameters, fits comfortably
        in 16 GB VRAM with batch_size=64 and sequence length 30+7.

    Args:
        num_features:  Total encoder input features (9)
        hidden_size:   LSTM hidden dimension (default 256)
        num_layers:    Stacked LSTM layers in both encoder and decoder (default 2)
        target_days:   Forecast horizon (default 7)
        num_targets:   Variables to predict probabilistically (default 4)
        dropout:       Regularisation dropout between LSTM layers (default 0.2)
    """
    def __init__(
        self,
        num_features=NUM_FEATURES,
        hidden_size=256,
        num_layers=2,
        target_days=TARGET_DAYS,
        num_targets=NUM_TARGET_FEATURES,
        dropout=0.2,
    ):
        super().__init__()
        self.encoder = Encoder(num_features, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, num_layers, target_days, num_targets, dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, INPUT_DAYS, NUM_FEATURES) — normalised input window
        Returns:
            mu:    (batch, TARGET_DAYS, NUM_TARGET_FEATURES)
            sigma: (batch, TARGET_DAYS, NUM_TARGET_FEATURES)
        """
        h, c = self.encoder(x)
        mu, sigma = self.decoder(h, c)
        return mu, sigma

# =============================================================================
# SECTION 6: LOSS FUNCTION AND TRAINING LOOP
# =============================================================================

def gaussian_nll_loss(mu, sigma, target):
    """
    Gaussian negative log-likelihood, averaged over all elements.

    For each predicted (mu, sigma) and true value y:
        NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
            = log(sigma) + 0.5 * ((y - mu) / sigma)^2  + const

    Summed across all 4 target variables and 7 forecast steps, then
    averaged over the batch. The constant 0.5*log(2*pi) is omitted —
    it does not affect optimisation.

    Args:
        mu:     (batch, TARGET_DAYS, NUM_TARGET_FEATURES) — predicted means
        sigma:  (batch, TARGET_DAYS, NUM_TARGET_FEATURES) — predicted std devs (> 0)
        target: (batch, TARGET_DAYS, NUM_TARGET_FEATURES) — ground truth values

    Returns:
        Scalar loss value.
    """
    variance = sigma ** 2
    return 0.5 * torch.mean(torch.log(variance) + (target - mu) ** 2 / variance)


def train_one_epoch(model, loader, optimiser, device):
    """
    Run one full pass over the training set.

    Args:
        model:     ProbabilisticForecaster
        loader:    training DataLoader
        optimiser: Adam optimiser
        device:    torch.device

    Returns:
        Mean NLL loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Extract only the 4 target variables from the full 9-feature target
        y_targets = y[:, :, TARGET_FEATURE_INDICES]   # (batch, TARGET_DAYS, 4)

        optimiser.zero_grad()
        mu, sigma = model(x)
        loss = gaussian_nll_loss(mu, sigma, y_targets)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in deep LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluate the model on a validation or test DataLoader.

    No gradients are computed. Returns mean NLL loss.

    Args:
        model:  ProbabilisticForecaster
        loader: DataLoader (val or test)
        device: torch.device

    Returns:
        Mean NLL loss over all batches.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_targets = y[:, :, TARGET_FEATURE_INDICES]
            mu, sigma = model(x)
            loss = gaussian_nll_loss(mu, sigma, y_targets)
            total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader, device, weights_path):
    """
    Full training loop with early stopping on validation NLL.

    Trains for up to EPOCHS epochs. After each epoch, evaluates on the
    validation set. Saves the best model weights whenever validation loss
    improves. Stops early if validation loss has not improved for
    PATIENCE consecutive epochs.

    Args:
        model:        ProbabilisticForecaster
        train_loader: training DataLoader
        val_loader:   validation DataLoader
        device:       torch.device
        weights_path: path to save best model .pt file

    Returns:
        history: dict with keys "train_loss" and "val_loss" (lists of floats)
    """
    PATIENCE = 10

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Reduce LR by 0.5 when val loss plateaus for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n[TRAIN] Starting training for up to {EPOCHS} epochs "
          f"(early stopping patience={PATIENCE})")
    print(f"[TRAIN] Device: {device}")
    print(f"[TRAIN] Best weights will be saved to: {weights_path}\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), weights_path)
            marker = " *"
        else:
            epochs_without_improvement += 1
            marker = ""

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train NLL: {train_loss:.4f} | "
            f"Val NLL: {val_loss:.4f}"
            f"{marker}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"\n[TRAIN] Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    print(f"\n[TRAIN] Done. Best val NLL: {best_val_loss:.4f} — weights saved to {weights_path}")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProbabilisticForecaster()
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] ProbabilisticForecaster — {num_params:,} trainable parameters")
    print(f"[MODEL] Device: {device}")

    # --- Step 6: Train ---
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    weights_path = os.path.join(WEIGHTS_DIR, "best_model.pt")
    history = train(model, train_loader, val_loader, device, weights_path)
    history_path = os.path.join(WEIGHTS_DIR, "loss_history.pt")
    torch.save(history, history_path)
    print(f"[TRAIN] Loss history saved to {history_path}")

    # --- Step 7: Evaluate on test set ---
    print("\n[EVAL] Loading best weights for test evaluation...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    test_nll = validate(model, test_loader, device)
    print(f"[EVAL] Test NLL: {test_nll:.4f}")

    # --- Step 8: Save normalisation stats alongside weights ---
    stats_path = os.path.join(WEIGHTS_DIR, "normalisation_stats.pt")
    torch.save({"mean": train_mean, "std": train_std}, stats_path)
    print(f"[EVAL] Normalisation stats saved to {stats_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()