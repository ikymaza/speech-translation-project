import pandas as pd

df = pd.read_csv("experiment/results_jalur1_wer_bleu.csv")

print("=== 5 sampel acak: ground truth vs hasil Whisper ===\n")
for _, row in df.head(5).iterrows():
    print(f"File: {row['file']}")
    print(f"  Ground truth : {row['ground_truth_id']}")
    print(f"  Predicted ID : {row['predicted_id']}")
    print(f"  WER: {row['wer']:.3f}\n")

print("\n=== 5 sampel dengan WER TERTINGGI ===\n")
for _, row in df.sort_values("wer", ascending=False).head(5).iterrows():
    print(f"File: {row['file']}")
    print(f"  Ground truth : {row['ground_truth_id']}")
    print(f"  Predicted ID : {row['predicted_id']}")
    print(f"  WER: {row['wer']:.3f}\n")