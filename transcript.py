import whisper
import csv
import os;
import warnings
from tqdm import tqdm
import jiwer 

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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#can add other supported formates and can also write code to convert formates  which is not needed in our dataset
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
                ground_truth = gt_file.read()
                predicted_text = result["text"] 
                #Word Error Rate calculation
                error = jiwer.wer(ground_truth, 
                            predicted_text,
                            truth_transform=transforms,
                            hypothesis_transform=transforms,)
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
audio_dir = "dataset/a/"
transcriptions = transcribe_with_whisper(audio_dir)
print("Transcriptions : ", transcriptions)

# Calculate overall accuracy,
# Treat negative accuracies as 0
adjusted_accuracies = [entry["accuracy"] if entry["accuracy"] >= 0 else 0 for entry in transcriptions]

# Calculate overall accuracy
if adjusted_accuracies:
    overall_accuracy = sum(adjusted_accuracies) / len(adjusted_accuracies)
    print(f"\nOverall Accuracy (adjusted): {overall_accuracy:.2f}%")

# Save the transcriptions to CSV
save_transcriptions_to_csv(transcriptions)
print("Transcriptions saved to baseTranscriptions.csv")