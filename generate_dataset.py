#!/usr/bin/env python3
"""
=====================================================================
Solar MPPT Dataset Generator
Based on Sagarika311/Maximum-Power-Point-Tracking Architecture & Physics
=====================================================================
Generates realistic solar PV operating data for training ANN MPPT models.
Models 50W Solar Panel (18.0V Vmpp, 2.78A Impp, 21.6V Voc, 2.92A Isc)
with Boost Converter Impedance Matching: Rin = Rload * (1 - D)^2.
=====================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd

def generate_mppt_dataset(num_samples=5000, r_load=15.0, seed=42):
    np.random.seed(seed)
    
    # 1. Environmental Sweeps
    # Irradiance: 100 to 1000 W/m² (uniform random + diurnal cluster)
    g_random = np.random.uniform(100, 1000, num_samples)
    
    # Temperature: 15°C to 65°C
    # Cell temp typically increases with high irradiance: T_cell ≈ T_amb + (NOCT - 20) * (G / 800)
    t_ambient = np.random.uniform(15, 45, num_samples)
    t_cell = t_ambient + 0.025 * g_random + np.random.normal(0, 2.0, num_samples)
    t_cell = np.clip(t_cell, 14.0, 65.0)

    # 2. 50W Solar Panel Physics (Single-Diode Model Characteristics)
    # Vmpp decreases with temperature (-0.08 V/°C), logarithmic dependence on G
    v_mpp = 18.0 - 0.082 * (t_cell - 25.0) + 0.52 * np.log(np.maximum(g_random / 1000.0, 0.05))
    v_mpp = np.clip(v_mpp, 12.0, 20.5)

    # Impp is directly proportional to Irradiance, slight positive temp coefficient (+0.05%/°C)
    i_mpp = 2.78 * (g_random / 1000.0) * (1.0 + 0.0005 * (t_cell - 25.0))
    i_mpp = np.clip(i_mpp, 0.25, 3.05)

    # Power at Maximum Power Point (Pmpp = Vmpp * Impp)
    p_mpp = v_mpp * i_mpp

    # 3. Previous Power (As in Sagarika's model: last measured step with noise / tracking deviation)
    # Simulates tracking step perturbation from 88% to 102% of Pmpp
    previous_power = p_mpp * np.random.uniform(0.88, 1.02, num_samples) + np.random.normal(0, 0.5, num_samples)
    previous_power = np.maximum(previous_power, 1.0)

    # 4. Boost Converter Optimal Duty Cycle Calculation
    # Matching solar panel resistance Rmpp = Vmpp / Impp to load R_load:
    # Rin = Rload * (1 - D)^2 => D = 1 - sqrt(Rmpp / Rload)
    r_mpp = v_mpp / i_mpp
    
    # Clamp ratio for physical boost operation (Rin <= Rload)
    ratio = np.clip(r_mpp / r_load, 0.02, 0.95)
    optimal_duty = 1.0 - np.sqrt(ratio)
    
    # Enforce minimum hardware clamp at 10% (0.10) and upper limit 85% (0.85)
    optimal_duty = np.clip(optimal_duty, 0.10, 0.85)

    # Add realistic sensor noise (quantization and ADC ripple)
    v_pv_measured = v_mpp + np.random.normal(0, 0.05, num_samples)
    i_pv_measured = i_mpp + np.random.normal(0, 0.015, num_samples)

    # Construct DataFrames
    # Full dataset with standard 4-sensor MPPT names matching notebook expectations
    df_full = pd.DataFrame({
        'Irradiance': np.round(g_random, 1),
        'Temperature': np.round(t_cell, 1),
        'V_pv': np.round(v_pv_measured, 3),
        'I_pv': np.round(i_pv_measured, 3),
        'previous_power': np.round(previous_power, 2),
        'Optimal_Duty': np.round(optimal_duty, 4),
        'optimal_duty_cycle': np.round(optimal_duty, 4)  # Backward compatibility alias
    })

    # Sagarika GitHub Schema (irradiance, temperature, previous_power, optimal_duty_cycle)
    df_sagarika = pd.DataFrame({
        'irradiance': np.round(g_random, 1),
        'temperature': np.round(t_cell, 1),
        'previous_power': np.round(previous_power, 2),
        'optimal_duty_cycle': np.round(optimal_duty, 4)
    })

    return df_full, df_sagarika

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Solar MPPT Dataset based on Sagarika311 model')
    parser.add_argument('--samples', type=int, default=5000, help='Number of rows to generate (default: 5000)')
    parser.add_argument('--rload', type=float, default=15.0, help='Boost load resistance in Ohms (default: 15.0)')
    args = parser.parse_args()

    print(f"Generating {args.samples} samples for 50W Solar Panel (Load: {args.rload} Ω)...")
    df_full, df_sagarika = generate_mppt_dataset(num_samples=args.samples, r_load=args.rload)

    # Save to disk
    full_path = 'mppt_generated_full_dataset.csv'
    sagarika_path = 'pv_data.csv'

    df_full.to_csv(full_path, index=False)
    df_sagarika.to_csv(sagarika_path, index=False)

    print(f"\n[SUCCESS] Datasets generated successfully!")
    print(f"1. Sagarika format saved to: '{sagarika_path}' (Shape: {df_sagarika.shape})")
    print(f"   Columns: {list(df_sagarika.columns)}")
    print(f"2. Full 6-feature dataset saved to: '{full_path}' (Shape: {df_full.shape})")
    print(f"   Columns: {list(df_full.columns)}\n")

    print("Preview of Sagarika Format (pv_data.csv):")
    print(df_sagarika.head())
