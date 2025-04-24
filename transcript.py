import whisper
import csv
import os;
import warnings
from tqdm import tqdm
from jiwer import wer


# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#can add other supported formates and can also write code to convert formates by using  which is not needed in our dataset
supported_formats = (".mp3") 

def transcribe_with_whisper(audio_dir):
    transcriptions = []
    files = [f for f in os.listdir(audio_dir) if f.endswith(supported_formats)]

    for filename in tqdm(files, desc="Transcribing audio files"):
        file_path = os.path.join(audio_dir, filename)
        
        # Load the Whisper model
        model = whisper.load_model("base")  #can change "base" to "large" for higher accuracy; 'base' is faster; 'large' gives better accuracy
        
        # Perform transcription
        result = model.transcribe(file_path)

        # Get ground truth from corresponding .txt file
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(audio_dir, txt_filename)
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as gt_file:
                ground_truth = gt_file.read().strip().lower() 
                predicted_text = result["text"].strip().replace('.', '').lower() #remove period because our dataset ground truth dosenot contains period and it was reducing the accuracy 
                #Word Error Rate calculation
                error = wer(ground_truth, predicted_text)
                #calculate accuracy
                accuracy = (1 - error) * 100
        else:
            ground_truth = ""
            accuracy = 0.0

        transcriptions.append({
            "filename": filename,
            "transcription": predicted_text,
            "ground_truth": ground_truth,
            "accuracy": round(accuracy, 2)
        })
    
    return transcriptions

def save_transcriptions_to_csv(transcriptions, output_file="baseTranscriptions.csv"):
    # Open the CSV file and write the transcriptions
     with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Transcription", "Ground Truth", "Accuracy (%)"])
        for entry in transcriptions:
            writer.writerow([entry["filename"], entry["transcription"], entry["ground_truth"], entry["accuracy"]])

# Run transcription on the audio datasets
audio_dir = "dataset/b/"
transcriptions = transcribe_with_whisper(audio_dir)

# Calculate overall accuracy, excluding invalid/negative values
valid_accuracies = [entry["accuracy"] for entry in transcriptions if entry["accuracy"] >= 0]

# Calculate overall accuracy
if valid_accuracies:
    overall_accuracy = sum(valid_accuracies) / len(valid_accuracies)
    print(f"\n✅ Overall Accuracy (filtered): {overall_accuracy:.2f}%")

# Save the transcriptions to CSV
save_transcriptions_to_csv(transcriptions)
print("Transcriptions saved to baseTranscriptions.csv")