# cli_runner.py

import argparse
import os
import torch
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice.se_extractor import get_se_from_audio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref_audio", required=True, help="Path to reference .wav file")
    parser.add_argument("--output_dir", required=True, help="Directory to save result")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_base = 'checkpoints/base_speakers/EN'
    ckpt_converter = 'checkpoints/converter'

    # Load models
    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    # Extract speaker embedding from your reference audio
    source_se = get_se_from_audio(args.ref_audio, tone_converter, device)

    # Inference
    print("🧠 Synthesizing speech...")
    save_path = os.path.join(args.output_dir, "ai_voice.wav")
    base_speaker_tts.tts(args.text, source_se, save_path)
    print(f"✅ Voice clone saved at: {save_path}")

if __name__ == "__main__":
    main()