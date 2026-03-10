"""
Fetch daily temperature data (min, max, mean) from Open-Meteo Historical Weather API
for the Sul de Minas coffee region in Minas Gerais, Brazil.

=== IMPORTANT: Coordinates ===
Sul de Minas (-21.25, -45.0) is the heart of Brazilian Arabica coffee production

Period: 15 years (2010-01-01 to 2024-12-31) = ~5,478 daily samples

"""

import urllib.request
import json
import csv
import os
import sys

# === Configuration ===
LATITUDE = -21.25
LONGITUDE = -45.0
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
TIMEZONE = "America/Sao_Paulo"
OUTPUT_CSV = "minas_gerais_coffee_temperature.csv"

# Daily variables to fetch
# temperature_2m_mean  = daily mean air temperature at 2m height (°C)
# temperature_2m_max   = daily maximum temperature (°C)
# temperature_2m_min   = daily minimum temperature (°C)
DAILY_VARS = "temperature_2m_mean,temperature_2m_max,temperature_2m_min"

# === Build API URL ===
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
url = (
    f"{BASE_URL}"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&start_date={START_DATE}"
    f"&end_date={END_DATE}"
    f"&daily={DAILY_VARS}"
    f"&timezone={TIMEZONE}"
)

print("=" * 60)
print("Open-Meteo Historical Weather Data Fetcher")
print("=" * 60)
print(f"Region:      Sul de Minas, Minas Gerais, Brazil")
print(f"Coordinates: {LATITUDE}, {LONGITUDE}")
print(f"Period:      {START_DATE} to {END_DATE}")
print(f"Variables:   Daily mean, max, min temperature (°C)")
print()
print("Browser download URL:")
print(url)
print()

# === Fetch data ===
print("Fetching data from Open-Meteo API...")
try:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Python/CourseworkProject")
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read().decode()
        data = json.loads(raw)
except Exception as e:
    print(f"\nError fetching data: {e}")
    print("\nTry downloading manually using URL above")

# === Check for API errors ===
if "error" in data and data["error"]:
    print(f"API Error: {data.get('reason', 'Unknown error')}")
    sys.exit(1)

# === Parse response ===
daily = data["daily"]
dates = daily["time"]
temp_mean = daily["temperature_2m_mean"]
temp_max = daily["temperature_2m_max"]
temp_min = daily["temperature_2m_min"]

print(f"\nRetrieved {len(dates)} daily records")
print(f"Date range: {dates[0]} to {dates[-1]}")

# === Count missing values ===
missing_mean = sum(1 for v in temp_mean if v is None)
missing_max = sum(1 for v in temp_max if v is None)
missing_min = sum(1 for v in temp_min if v is None)
print(f"Missing values: mean={missing_mean}, max={missing_max}, min={missing_min}")

# === Save to CSV ===
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_CSV)
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "temp_mean_c", "temp_max_c", "temp_min_c"])
    for i in range(len(dates)):
        writer.writerow([dates[i], temp_mean[i], temp_max[i], temp_min[i]])

print(f"\nSaved to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")

# === Summary statistics ===
valid_mean = [v for v in temp_mean if v is not None]
valid_max = [v for v in temp_max if v is not None]
valid_min = [v for v in temp_min if v is not None]

print(f"\n{'=' * 60}")
print(f"Temperature Summary (°C)")
print(f"{'=' * 60}")
print(f"{'Variable':<12} {'Avg':>8} {'Min':>8} {'Max':>8}")
print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8}")
print(f"{'Mean temp':<12} {sum(valid_mean)/len(valid_mean):>8.1f} {min(valid_mean):>8.1f} {max(valid_mean):>8.1f}")
print(f"{'Max temp':<12} {sum(valid_max)/len(valid_max):>8.1f} {min(valid_max):>8.1f} {max(valid_max):>8.1f}")
print(f"{'Min temp':<12} {sum(valid_min)/len(valid_min):>8.1f} {min(valid_min):>8.1f} {max(valid_min):>8.1f}")

# === Agricultural relevance ===
frost_days = sum(1 for v in valid_min if v < 2.0)
heat_days = sum(1 for v in valid_max if v > 30.0)
optimal_days = sum(1 for m in valid_mean if 18.0 <= m <= 24.0)

print(f"\n{'=' * 60}")
print(f"Coffee-Relevant Statistics")
print(f"{'=' * 60}")
print(f"Frost-risk days (min < 2°C):          {frost_days:>5} ({frost_days/len(valid_min)*100:.1f}%)")
print(f"Heat-stress days (max > 30°C):        {heat_days:>5} ({heat_days/len(valid_max)*100:.1f}%)")
print(f"Optimal Arabica days (mean 18-24°C):  {optimal_days:>5} ({optimal_days/len(valid_mean)*100:.1f}%)")
print(f"\nArabica coffee thrives at 18-24°C mean temperature.")
print(f"Frost (< 2°C) can kill plants; sustained heat (> 30°C) reduces quality.")
