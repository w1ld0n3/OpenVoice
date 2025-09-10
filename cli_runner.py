# cli_runner.py (v2 - compatible with latest OpenVoice)

import argparse
import os
import torch
import torchaudio
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

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

    print("🧬 Extracting speaker embedding (using convert_audio_to_embed)...")

    # Load and preprocess audio
    wav, sr = torchaudio.load(args.ref_audio)
    wav = wav.mean(dim=0, keepdim=True)  # ensure mono
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    wav = wav.to(device)

    # Convert to speaker embedding
    speaker_embed = tone_converter.convert_audio_to_embed(wav)

    # Generate speech
    print("🗣️ Synthesizing speech...")
    output_path = os.path.join(args.output_dir, "ai_voice.wav")
    base_speaker_tts.tts(args.text, speaker_embed, output_path)

    print(f"✅ Voice clone saved at: {output_path}")

if __name__ == "__main__":
    main()
