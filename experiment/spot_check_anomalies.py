import pandas as pd

df = pd.read_csv("experiment/results_jalur1_wer_bleu.csv")

print("=== 5 sampel dengan WER tertinggi (>1.0) ===")
high_wer = df[df["wer"] > 1.0].sort_values("wer", ascending=False)
for _, row in high_wer.head(5).iterrows():
    print(f"\nFile: {row['file']}")
    print(f"  Ground truth : {row['ground_truth_id']}")
    print(f"  Predicted ID : {row['predicted_id']}")
    print(f"  WER: {row['wer']:.3f}")

print("\n\n=== 5 sampel dengan BLEU tertinggi (>85) ===")
high_bleu = df[df["bleu"] > 85].sort_values("bleu", ascending=False)
for _, row in high_bleu.head(5).iterrows():
    print(f"\nFile: {row['file']}")
    print(f"  Reference EN : {row['reference_en']}")
    print(f"  Predicted EN : {row['predicted_en']}")
    print(f"  BLEU: {row['bleu']:.2f}")