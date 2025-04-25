# Whisper ASR Comparison: Baseline vs Modified Whisper (Goodnight Moon 1st Place)

# Step 0: Install required packages (run in shell or notebook cell)
# !pip install git+https://github.com/drscotthawley/whisper-timestamped.git
# !pip install jiwer pandas matplotlib

import os
import whisper
import pandas as pd
import matplotlib.pyplot as plt
import jiwer 
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load Models
baseline_model = whisper.load_model("base")
finetuned_model = whisper.load_model("base")

# WER Transformation
transforms = jiwer.Compose(
    [
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

# Load Data from CSV
input_csv = "baseTranscriptions.csv"  # Replace with your actual CSV path
data = pd.read_csv(input_csv)

# Initialize results
results = []

# Directory containing audio files
audio_dir = "dataset/a/"  # Replace with your directory

# Evaluate each file
for index, row in data.iterrows():
    filename = row["Filename"]
    ground_truth = row["Ground Truth"]
    filepath = os.path.join(audio_dir, filename)
    print(filepath)
    # Skip if file not found
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue

    try:
        baseline_result = baseline_model.transcribe(filepath)
        modified_result = finetuned_model.transcribe(filepath)

        gt_clean = ground_truth
        baseline_clean = baseline_result["text"]
        modified_clean = modified_result["text"]

        baseline_acc = max(round((1 - jiwer.wer(gt_clean, baseline_clean, truth_transform=transforms,
                                                        hypothesis_transform=transforms,)) * 100, 2), 0)
        modified_acc = max(round((1 - jiwer.wer(gt_clean, modified_clean,truth_transform=transforms,
                                                        hypothesis_transform=transforms,)) * 100, 2),0)

        results.append({
            "Filename": filename,
            "Ground Truth": ground_truth,
            "Baseline": baseline_result["text"],
            "Modified": modified_result["text"],
            "Baseline Accuracy": baseline_acc,
            "Modified Accuracy": modified_acc
        })

    except Exception as e:
        print(f" Error processing {filename}: {str(e)}")

# Save Results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("comparison_results.csv", index=False)

# Average Accuracies
baseline_avg = results_df["Baseline Accuracy"].mean()
modified_avg = results_df["Modified Accuracy"].mean()

print(f"\n📊 Baseline Avg Accuracy: {baseline_avg:.2f}%")
print(f"📊 Modified Avg Accuracy: {modified_avg:.2f}%")

# Plot Comparison
plt.figure(figsize=(6, 4))
plt.bar(["Baseline", "Modified"], [baseline_avg, modified_avg], color=["#1f77b4", "#2ca02c"])
plt.ylabel("Average Accuracy (%)")
plt.title("Whisper Model Accuracy Comparison")
plt.ylim(0, 100)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("whisper_accuracy_comparison.png")
plt.show()
