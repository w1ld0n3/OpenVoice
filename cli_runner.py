# cli_runner.py

import argparse
import os
import torch
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import torchaudio

def extract_se(audio_path: str, converter: ToneColorConverter, device: str):
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(dim=0, keepdim=True)  # make mono
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    se = converter.get_se(wav.to(device))
    return se

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_base = 'checkpoints/base_speakers/EN'
    ckpt_converter = 'checkpoints/converter'

    print("🧠 Loading models...")
    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    print("🧬 Extracting speaker embedding...")
    source_se = extract_se(args.ref_audio, tone_converter, device)

    print("🗣️ Synthesizing voice...")
    save_path = os.path.join(args.output_dir, "ai_voice.wav")
    base_speaker_tts.tts(args.text, source_se, save_path)

    print(f"✅ Voice clone saved at: {save_path}")

if __name__ == "__main__":
    main()
