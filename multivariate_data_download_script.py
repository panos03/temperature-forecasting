"""
ERA5 Data Download & Processing Script
=======================================
Downloads hourly weather data for a single grid point near Varginha,
Minas Gerais (the heart of Brazil's Arabica coffee belt) and processes
it into daily aggregates.

BEFORE RUNNING:
1. Create a free account at https://cds.climate.copernicus.eu/
2. Go to your profile and copy your API key
3. Create a file at ~/.cdsapirc with:
       url: https://cds.climate.copernicus.eu/api
       key: YOUR-API-KEY-HERE
4. Install dependencies:
       pip install cdsapi pandas numpy
"""

import cdsapi
import numpy as np
import pandas as pd
import zipfile
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Single grid point near Varginha, southern Minas Gerais
LATITUDE = -21.5
LONGITUDE = -45.5

# Date range: 2004 to 2024
DATE_RANGE = "2004-01-01/2024-12-31"

# Variables to download from ERA5 single-levels timeseries
VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    'total_precipitation',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_pressure',
]

# Output directories
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# =============================================================================
# STEP 1: DOWNLOAD
# =============================================================================

def download_era5():
    """
    Download ERA5 timeseries data for the entire date range at once.
    The CDS always returns a zip file, so we extract the CSV from it.
    """
    client = cdsapi.Client()

    dataset = "reanalysis-era5-single-levels-timeseries"
    request = {
        "variable": VARIABLES,
        "location": {"longitude": LONGITUDE, "latitude": LATITUDE},
        "date": [DATE_RANGE],
        "data_format": "csv"
    }

    csv_path = os.path.join(RAW_DIR, 'era5_minas_gerais_hourly.csv')

    # Skip if we already have the extracted CSV
    if os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} already exists")
        return

    download_path = os.path.join(RAW_DIR, 'era5_download.zip')

    print(f"[DOWNLOAD] Requesting ERA5 data for {DATE_RANGE}...")

    try:
        client.retrieve(dataset, request).download(download_path)
        print(f"[DONE] Downloaded to {download_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download data: {e}")
        return

    # CDS always wraps output in a zip — extract the CSV
    print("[EXTRACT] Unzipping...")
    with zipfile.ZipFile(download_path, 'r') as zf:
        csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
        if not csv_files:
            print(f"[ERROR] No CSV found in zip. Contents: {zf.namelist()}")
            return
        zf.extract(csv_files[0], RAW_DIR)
        extracted = os.path.join(RAW_DIR, csv_files[0])
        os.rename(extracted, csv_path)

    os.remove(download_path)
    print(f"[DONE] Extracted to {csv_path}")


# =============================================================================
# STEP 2: PROCESS INTO DAILY AGGREGATES
# =============================================================================

def compute_relative_humidity(temp_k, dewpoint_k):
    """
    Compute relative humidity (%) from temperature and dewpoint
    using the Magnus formula approximation.
    """
    temp_c = temp_k - 273.15
    dewpoint_c = dewpoint_k - 273.15

    a = 17.625
    b = 243.04

    rh = 100.0 * np.exp((a * dewpoint_c) / (b + dewpoint_c)) / \
                  np.exp((a * temp_c) / (b + temp_c))

    return np.clip(rh, 0, 100)


def process_to_daily():
    """
    Load the hourly CSV, compute derived variables, and aggregate to daily.

    ERA5 CSV columns:
        valid_time, u10, v10, d2m, t2m, sp, tp, latitude, longitude
    """
    csv_path = os.path.join(RAW_DIR, 'era5_minas_gerais_hourly.csv')

    if not os.path.exists(csv_path):
        print("[ERROR] CSV file not found. Run download first.")
        return

    print("[PROCESS] Loading hourly CSV...")
    df = pd.read_csv(csv_path, parse_dates=['valid_time'], index_col='valid_time')
    df = df.sort_index()

    print(f"[PROCESS] Loaded {len(df)} hourly rows")
    print(f"[PROCESS] Date range: {df.index[0]} to {df.index[-1]}")
    print(f"[PROCESS] Columns: {list(df.columns)}")

    # ----- Compute derived variables -----
    print("[PROCESS] Computing derived variables...")

    # Temperature: Kelvin -> Celsius
    temp_c = df['t2m'] - 273.15

    # Relative humidity from temperature and dewpoint
    rh = compute_relative_humidity(df['t2m'].values, df['d2m'].values)

    # Wind speed from u and v components
    wind_speed = np.sqrt(df['u10']**2 + df['v10']**2)

    # Pressure: Pa -> hPa
    pressure_hpa = df['sp'] / 100.0

    # Precipitation (metres per hour)
    precip = df['tp']

    # ----- Daily aggregation -----
    print("[PROCESS] Aggregating to daily values...")

    daily = pd.DataFrame({
        'temperature_mean_c': temp_c.resample('1D').mean(),
        'temperature_min_c': temp_c.resample('1D').min(),
        'temperature_max_c': temp_c.resample('1D').max(),
        'relative_humidity_mean_pct': pd.Series(rh, index=df.index).resample('1D').mean(),
        'total_precipitation_mm': precip.resample('1D').sum() * 1000,  # m -> mm
        'wind_speed_mean_ms': wind_speed.resample('1D').mean(),
        'surface_pressure_mean_hpa': pressure_hpa.resample('1D').mean(),
    })

    daily.index.name = 'date'
    daily = daily.dropna(how='all')

    # ----- Save -----
    output_path = os.path.join(PROCESSED_DIR, 'minas_gerais_daily_weather.csv')
    daily.to_csv(output_path)

    print(f"\n[DONE] Saved daily data to {output_path}")
    print(f"       Shape: {daily.shape}")
    print(f"       Date range: {daily.index[0]} to {daily.index[-1]}")
    print(f"\nFirst few rows:")
    print(daily.head())
    print(f"\nBasic statistics:")
    print(daily.describe())

    return daily


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ERA5 Download Script — Minas Gerais Coffee Belt")
    print("=" * 60)
    print(f"Location: {LATITUDE}°S, {LONGITUDE}°W (near Varginha)")
    print(f"Date range: {DATE_RANGE}")
    print(f"Variables: {', '.join(VARIABLES)}")
    print("=" * 60)

    # Step 1: Download and extract
    download_era5()

    # Step 2: Process to daily CSV
    daily_data = process_to_daily()