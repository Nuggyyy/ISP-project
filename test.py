from datasets import load_dataset, Audio

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

print(dataset)
for key in dataset:
    print(dataset[key][0])

for sample in dataset["clipped"]:
    print(sample)
    print(sample["transcription"])
