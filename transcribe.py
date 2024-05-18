import argparse
import os
import time
from lightning_whisper_mlx import LightningWhisperMLX

def transcribe_audio(audio_file, model, quant, output_path):
    valid_models = ["tiny", "small", "distil-small.en", "base", "medium", "distil-medium.en", "large", "large-v2", "distil-large-v2", "large-v3", "distil-large-v3"]
    valid_quant = [None, "4bit", "8bit"]

    if model not in valid_models:
        print(f"Error: '{model}' is not a valid Whisper model. Please choose one of the following: {', '.join(valid_models)}")
        return
    if quant not in valid_quant:
        print(f"Error: '{quant}' is not a valid quantization option. Please choose one of the following: {', '.join(valid_quant)}")
        return

    start_time = time.time()
    whisper = LightningWhisperMLX(model=model, batch_size=12, quant=quant)
    load_time = time.time() - start_time
    print(f"Whisper model loaded in {load_time:.2f} seconds.")

    start_time = time.time()
    text = whisper.transcribe(audio_path=audio_file)['text']
    transcribe_time = time.time() - start_time
    print(f"Audio transcribed in {transcribe_time:.2f} seconds.")

    audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = os.path.join(output_path, f"{audio_filename}_{model}_{quant}_transcription.txt")
    os.makedirs(output_path, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Transcription saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio file using Whisper")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--model", default="tiny", help="Whisper model to use (default: tiny). Available models: tiny, small, distil-small.en, base, medium, distil-medium.en, large, large-v2, distil-large-v2, large-v3, distil-large-v3.")
    parser.add_argument("-q", "--quant", default=None, help="Quantization option (None, '4bit', '8bit')")
    parser.add_argument("-o", "--output", default=".", help="Output directory for the transcription file (default: current directory)")
    args = parser.parse_args()
    transcribe_audio(args.audio_file, args.model, args.quant, args.output)
