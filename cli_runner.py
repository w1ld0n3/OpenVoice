# cli_runner.py (verified against latest OpenVoice)

import argparse
import os
import torch
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

    print("🧬 Extracting speaker style vector...")
    style_vector = tone_converter.get_style_vector(args.ref_audio)

    print("🗣️ Synthesizing speech...")
    output_path = os.path.join(args.output_dir, "ai_voice.wav")
    base_speaker_tts.tts(args.text, style_vector, output_path)

    print(f"✅ Voice clone saved at: {output_path}")

if __name__ == "__main__":
    main()
