import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

def load_wav2vec2_asr_model(device="cpu"):
    """
    Load the Wav2Vec2 ASR model on the specified device ('cpu' or 'cuda').
    Default is 'cpu'.
    """
    model = WAV2VEC2_ASR_BASE_960H.get_model().to(device)
    model.eval()
    return model

def transcribe_audio(model, audio_path, device="cpu"):
    """
    Transcribe audio using the given model and device ('cpu' or 'cuda').
    Default is 'cpu'.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != WAV2VEC2_ASR_BASE_960H.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, WAV2VEC2_ASR_BASE_960H.sample_rate)

    waveform = waveform.to(device)
    model = model.to(device)

    with torch.no_grad():
        emission, _ = model(waveform)

    decoder = GreedyCTCDecoder(labels=WAV2VEC2_ASR_BASE_960H.get_labels())
    transcript = decoder(emission[0])
    return transcript