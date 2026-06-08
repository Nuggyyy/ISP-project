from transformers import AutoTokenizer
from datasets import load_dataset
from jiwer import wer

tokenizer = AutoTokenizer.from_pretrained("openai/whisper-large-v3")

data_files = {
    "clipped": "./distorted_data/clipped/clipped_transcriptions.csv",
    "downsampled": "./distorted_data/downsampled/downsampled_transcriptions.csv",
    "normal": "./distorted_data/normal/normal_transcriptions.csv",
    "whitenoise": "./distorted_data/whitenoise/whitenoise_transcriptions.csv",
}
reference_dataset = load_dataset("csv", data_files=data_files, column_names=["id", "transcription"])

data_files = {
    "clipped": "./human_transcription_csv/clipped.csv",
    "downsampled": "./human_transcription_csv/downsampled.csv",
    "normal": "./human_transcription_csv/normal.csv",
    "whitenoise": "./human_transcription_csv/whitenoise.csv",
}
hypothesis_dataset = load_dataset("csv", data_files=data_files, column_names=["file", "transcription"])

clipped_references = sorted(reference_dataset["clipped"], key=lambda d: d['id'])
downsampled_references = sorted(reference_dataset["downsampled"], key=lambda d: d['id'])
normal_references = sorted(reference_dataset["normal"], key=lambda d: d['id'])
whitenoise_references = sorted(reference_dataset["whitenoise"], key=lambda d: d['id'])

clipped_references_norm = [tokenizer.normalize(sample["transcription"]) for sample in clipped_references]
downsampled_references_norm = [tokenizer.normalize(sample["transcription"]) for sample in downsampled_references]
normal_references_norm = [tokenizer.normalize(sample["transcription"]) for sample in normal_references]
whitenoise_references_norm = [tokenizer.normalize(sample["transcription"]) for sample in whitenoise_references]


clipped_hypotheses = sorted(hypothesis_dataset["clipped"], key=lambda d: d['file'])
downsampled_hypotheses = sorted(hypothesis_dataset["downsampled"], key=lambda d: d['file'])
normal_hypotheses = sorted(hypothesis_dataset["normal"], key=lambda d: d['file'])
whitenoise_hypotheses = sorted(hypothesis_dataset["whitenoise"], key=lambda d: d['file'])

clipped_hypotheses_norm = [tokenizer.normalize(sample["transcription"]) for sample in clipped_hypotheses]
downsampled_hypotheses_norm = [tokenizer.normalize(sample["transcription"]) for sample in downsampled_hypotheses]
normal_hypotheses_norm = [tokenizer.normalize(sample["transcription"]) for sample in normal_hypotheses]
whitenoise_hypotheses_norm = [tokenizer.normalize(sample["transcription"]) for sample in whitenoise_hypotheses]

print("WER clipped:", round(wer(clipped_references_norm, clipped_hypotheses_norm) * 100, 2), "%")
print("WER downsampled:", round(wer(downsampled_references_norm, downsampled_hypotheses_norm) * 100, 2), "%")
print("WER normal:", round(wer(normal_references_norm, normal_hypotheses_norm) * 100, 2), "%")
print("WER whitenoise:", round(wer(whitenoise_references_norm, whitenoise_hypotheses_norm) * 100, 2), "%")
