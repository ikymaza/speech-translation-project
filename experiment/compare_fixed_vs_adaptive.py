"""
Gabungkan hasil Jalur 2 (fixed) dan adaptive window jadi 1 tabel
perbandingan ringkas, siap dipakai sebagai tabel utama di Bab IV.
"""
import pandas as pd
from pathlib import Path

FIXED_PATH = Path("experiment/results_jalur2_fixed_window.csv")
ADAPTIVE_PATH = Path("experiment/results_adaptive_window.csv")
OUTPUT_PATH = Path("experiment/comparison_summary.csv")


def main():
    fixed = pd.read_csv(FIXED_PATH)
    adaptive = pd.read_csv(ADAPTIVE_PATH)

    # --- Ringkasan Fixed Window, per nilai r ---
    fixed_summary = fixed.groupby("r").agg(
        tingkat_gagal_persen=("status", lambda s: round((s == "FAILED").mean() * 100, 2)
                               if s.dtype == object else round((s != "ok").mean() * 100, 2)),
        avg_normalized_distance=("normalized_distance", "mean"),
        avg_rtf_partial=("rtf_partial", "mean"),
    ).reset_index()
    fixed_summary.insert(0, "mode", "fixed")
    fixed_summary = fixed_summary.rename(columns={"r": "parameter"})

    # --- Ringkasan Adaptive Window, per margin ---
    adaptive_summary = adaptive.groupby("margin").agg(
        tingkat_gagal_persen=("status", lambda s: round((s != "ok").mean() * 100, 2)),
        avg_normalized_distance=("normalized_distance", "mean"),
        avg_rtf_partial=("rtf_partial", "mean"),
    ).reset_index()
    adaptive_summary.insert(0, "mode", "adaptive")
    adaptive_summary = adaptive_summary.rename(columns={"margin": "parameter"})

    combined = pd.concat([fixed_summary, adaptive_summary], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print("Tabel perbandingan Fixed vs Adaptive Window:\n")
    print(combined.to_string(index=False))
    print(f"\nDisimpan ke: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()