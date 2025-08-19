import gradio as gr
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice, load_voices
import torchaudio

tts = TextToSpeech()

def synthesize(text, voice="tom", preset="fast"):
    voices = load_voices()
    if voice in voices:
        voice_samples, conditioning_latents = load_voice(voice)
    else:
        voice_samples, conditioning_latents = load_voice("tom")

    wav = tts.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset=preset
    )
    out_path = "output.wav"
    torchaudio.save(out_path, wav.squeeze(0).cpu(), 22050)
    return out_path

demo = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Textbox(label="Text", placeholder="Type your text here..."),
        gr.Textbox(label="Voice", value="tom"),
        gr.Dropdown(choices=["ultra_fast", "fast", "standard", "high_quality"], value="fast", label="Preset"),
    ],
    outputs=gr.Audio(type="filepath", label="Generated Audio"),
    title="Tortoise TTS – Gradio UI",
    description="Simple wrapper UI for Tortoise TTS"
)

if __name__ == "__main__":
    demo.launch()
