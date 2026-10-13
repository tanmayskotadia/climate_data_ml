"""
Multi-Source Climate Data Generator & Loader
Simulates realistic atmospheric & meteorological observations mirroring:
- ERA5 (Gridded Reanalysis)
- IMD (India Meteorological Department Ground Stations)
- NASA POWER (Satellite & Agroclimatology Assimilation)
- NOAA GSOD (Global Surface Summary of the Day)
Includes physical diurnal cycles, seasonal oscillation, autoregressive red noise, and multi-variable correlations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

def generate_climate_dataset(
    source: str = "ERA5",
    num_hours: int = 720,  # 30 days of hourly data by default
    start_date: str = "2024-01-01",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a physically realistic multi-variable climate time series.
    Variables:
      - temperature (°C): diurnal cycle + seasonal trend + AR(1) noise
      - relative_humidity (%): anti-correlated with temperature + daily cycle
      - wind_speed (m/s): Weibull-distributed fluctuating wind
      - solar_radiation (W/m^2): daylight half-sine curve modulated by clouds
      - precipitation (mm): intermittent zero-inflated Gamma process
    """
    np.random.seed(seed)
    timestamps = [pd.to_datetime(start_date) + timedelta(hours=i) for i in range(num_hours)]
    t = np.arange(num_hours)
    
    hour_of_day = np.array([ts.hour for ts in timestamps])
    day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps])

    # Source-specific base parameters
    source_params = {
        "ERA5": {"base_temp": 22.0, "diurnal_amp": 5.5, "noise_std": 0.6, "station_noise": 0.2},
        "IMD": {"base_temp": 28.5, "diurnal_amp": 7.0, "noise_std": 1.2, "station_noise": 0.8},
        "NASA POWER": {"base_temp": 24.0, "diurnal_amp": 6.0, "noise_std": 0.8, "station_noise": 0.4},
        "NOAA GSOD": {"base_temp": 18.0, "diurnal_amp": 4.5, "noise_std": 1.0, "station_noise": 0.5},
    }
    params = source_params.get(source, source_params["ERA5"])

    # 1. Temperature (°C)
    # Diurnal peak at ~14:00 (phase shift = -14)
    diurnal_temp = params["diurnal_amp"] * np.sin(2 * np.pi * (hour_of_day - 8) / 24.0)
    seasonal_temp = 4.0 * np.sin(2 * np.pi * day_of_year / 365.25)
    # Autoregressive red noise AR(1)
    ar_noise = np.zeros(num_hours)
    for i in range(1, num_hours):
        ar_noise[i] = 0.85 * ar_noise[i-1] + np.random.normal(0, params["noise_std"])
    temp = params["base_temp"] + diurnal_temp + seasonal_temp + ar_noise

    # 2. Relative Humidity (%)
    # Inversely correlated with temperature
    rh_base = 70.0 - (temp - params["base_temp"]) * 2.2
    rh_noise = np.random.normal(0, 3.5, num_hours)
    rh = np.clip(rh_base + rh_noise, 15.0, 99.0)

    # 3. Wind Speed (m/s)
    # Weibull style positive values
    base_wind = 3.5 + 1.2 * np.sin(2 * np.pi * (hour_of_day - 12) / 24.0)
    wind_noise = np.random.exponential(scale=1.5, size=num_hours)
    wind_speed = np.clip(base_wind + wind_noise, 0.2, 28.0)

    # 4. Solar Radiation (W/m^2)
    # Positive only during day hours (06:00 to 18:00)
    solar_rad = np.zeros(num_hours)
    for i, h in enumerate(hour_of_day):
        if 6 <= h <= 18:
            sun_phase = (h - 6) / 12.0 * np.pi
            cloud_factor = np.clip(1.0 - (rh[i] - 40.0) / 100.0 * 0.7 + np.random.normal(0, 0.08), 0.1, 1.0)
            solar_rad[i] = 950.0 * np.sin(sun_phase) * cloud_factor
    solar_rad = np.clip(solar_rad, 0.0, 1100.0)

    # 5. Precipitation (mm)
    # Intermittent: only when RH is high (> 80%) with a stochastic trigger
    precip = np.zeros(num_hours)
    rain_probability = np.where(rh > 78.0, (rh - 78.0) / 40.0, 0.0)
    for i in range(num_hours):
        if np.random.rand() < rain_probability[i]:
            precip[i] = np.random.gamma(shape=1.5, scale=4.0)
    precip = np.round(precip, 2)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.round(temp, 2),
        "relative_humidity": np.round(rh, 1),
        "wind_speed": np.round(wind_speed, 2),
        "solar_radiation": np.round(solar_rad, 1),
        "precipitation": precip,
        "source": source
    })
    return df
