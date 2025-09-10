import argparse
import os
import torch
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice import se_extractor

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
    base_speaker_tts = BaseSpeakerTTS(f"{ckpt_base}/config.json", device=device)
    base_speaker_tts.load_ckpt(f"{ckpt_base}/checkpoint.pth")

    tone_color_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
    tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

    print("🧬 Extracting target speaker embedding...")
    tgt_se, _ = se_extractor.get_se(
        args.ref_audio, tone_color_converter, target_dir='processed', vad=True
    )

    print("🗣️ Generating neutral speech...")
    neutral_audio_path = os.path.join(args.output_dir, "tmp.wav")
    base_speaker_tts.tts(
        text=args.text,
        speaker='default',
        output_path=neutral_audio_path,
        language='English',
        speed=1.0
    )

    print("🎭 Applying tone conversion (cloning)...")
    final_audio_path = os.path.join(args.output_dir, "ai_voice.wav")

    # You can optionally use a pre-computed default speaker embedding here:
    source_se = tone_color_converter.extract_se(args.ref_audio)  # or make a short neutral base voice

    tone_color_converter.convert(
        audio_src_path=neutral_audio_path,
        src_se=source_se,
        tgt_se=tgt_se,
        output_path=final_audio_path,
        message="@DigitalSelf"
    )

    print(f"✅ Final voice clone saved at: {final_audio_path}")

if __name__ == "__main__":
    main()
