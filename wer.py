import requests
import os
import numpy as np
import tqdm
#try:
#    import tensorflow  # required in Colab to avoid protobuf compatibility issues
#except ImportError:
#    pass

import torch
import pandas as pd
import whisper
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#from tqdm.notebook import tqdm

def postprocess(data):
    return data.tolist()

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        
        assert sample_rate == 16000
        # preprocessing
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
   
        return (mel, text)

dataset = LibriSpeech("test-clean")
loader = torch.utils.data.DataLoader(dataset, batch_size=16)

# predict without timestamps for short-form transcription
hypotheses = []
references = []

for mels, texts in tqdm.tqdm(loader):
    data = postprocess(mels)
    results = requests.post("http://34.126.164.20:8085/", json={"audio": data, "Model": "STT"}).json()["results"]
    hypotheses.extend([result["text"] for result in results])
    references.extend(texts)
data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

import jiwer
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()
data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
data["reference_clean"] = [normalizer(text) for text in data["reference"]]
wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

print(f"WER: {wer * 100:.2f} %")
