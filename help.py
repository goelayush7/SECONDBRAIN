import io
import wave
import simpleaudio as sa
from groq import Groq

def speak(text: str):
    client = Groq(api_key="gsk...")
    resp = client.audio.speech.create(
        model="playai-tts",
        voice="Aaliyah-PlayAI",
        response_format="wav",
        input=text,
    )
    buf = io.BytesIO()
    for chunk in resp.iter_bytes(chunk_size=1024):
        buf.write(chunk)
    buf.seek(0)

    # 2) Read WAV header/frames
    with wave.open(buf, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        num_channels = wf.getnchannels()
        samp_width   = wf.getsampwidth()
        framerate    = wf.getframerate()

    # 3) Play it
    play_obj = sa.WaveObject(frames, num_channels, samp_width, framerate).play()
    play_obj.wait_done()

if __name__ == "__main__":
    speak("Protein Sources:** Without protein powder, we’ll rely on paneer, tofu, soya chunks, lentils, dairy, nuts, and non-veg (dinner only on non-veg days).**Non-Veg Restriction:** Non-veg is limited to dinner on non-veg days (Sunday, Monday, Wednesday, Thursday, Friday). All other meals/snacks on those days are vegetarian. **Vegetarian Days:** Tuesday and Saturday are fully vegetarian.Hydration:** Drink at least 3 liters of water daily.Portion Adjustments:** Increase rice, roti, or snack sizes if you’re still hungry or need more calories.")  # change this text as you like
