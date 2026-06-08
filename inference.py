import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio
from jiwer import wer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=torch_dtype,
    device=device,
)

# Load the tab-separated transcription file (no header). Columns: id, transcription
data_files = {
    "clipped": "./distorted_data/clipped/clipped_transcriptions.csv",
    "downsampled": "./distorted_data/downsampled/downsampled_transcriptions.csv",
    "normal": "./distorted_data/normal/normal_transcriptions.csv",
    "whitenoise": "./distorted_data/whitenoise/whitenoise_transcriptions.csv",
}
dataset = load_dataset("csv", data_files=data_files, column_names=["id", "transcription"])

for split in dataset.keys():
    def add_audio(batch, split=split):
        ids = batch.get("id", batch.get("filename", []))
        batch["audio"] = [
            f"./distorted_data/{split}/{fname}.flac"
            for fname in ids
        ]
        return batch
    dataset[split] = dataset[split].map(add_audio, batched=True)
    dataset[split] = dataset[split].cast_column("audio", Audio(sampling_rate=16000))

references = {
    "clipped": [],
    "downsampled": [],
    "normal": [],
    "whitenoise": [],
}
hypotheses = {
    "clipped": [],
    "downsampled": [],
    "normal": [],
    "whitenoise": [],
}



for sample in dataset["clipped"]:
    references["clipped"].append(processor.tokenizer.normalize(sample["transcription"]))
    hypotheses["clipped"].append(processor.tokenizer.normalize(pipe(sample["audio"])["text"]))
for sample in dataset["downsampled"]:
    references["downsampled"].append(processor.tokenizer.normalize(sample["transcription"]))
    hypotheses["downsampled"].append(processor.tokenizer.normalize(pipe(sample["audio"])["text"]))
for sample in dataset["normal"]:
    references["normal"].append(processor.tokenizer.normalize(sample["transcription"]))
    hypotheses["normal"].append(processor.tokenizer.normalize(pipe(sample["audio"])["text"]))
for sample in dataset["whitenoise"]:
    references["whitenoise"].append(processor.tokenizer.normalize(sample["transcription"]))
    hypotheses["whitenoise"].append(processor.tokenizer.normalize(pipe(sample["audio"])["text"]))

print("WER clipped:", round(wer(references["clipped"], hypotheses["clipped"]) * 100, 2), "%")
print("WER downsampled:", round(wer(references["downsampled"], hypotheses["downsampled"]) * 100, 2), "%")
print("WER normal:", round(wer(references["normal"], hypotheses["normal"]) * 100, 2), "%")
print("WER whitenoise:", round(wer(references["whitenoise"], hypotheses["whitenoise"]) * 100, 2), "%")


#sample = dataset["downsampled"][0]["audio"]
#result = pipe(sample)
#normalized_result = processor.tokenizer.normalize(result["text"])
#normalized_reference = processor.tokenizer.normalize(dataset["downsampled"][0]["transcription"])
#print("result:", normalized_result)
#print("reference:", normalized_reference)
#print("WER:", wer(normalized_reference, normalized_result))
