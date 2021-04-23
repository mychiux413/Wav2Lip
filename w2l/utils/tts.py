from pydub import AudioSegment
from io import BytesIO, StringIO
import os
import pydub
from google.cloud import texttospeech


def create_client():
    return texttospeech.TextToSpeechClient()


def list_voices(client):
    voices_table = {}
    voices = {}
    for i, v in enumerate(sorted(client.list_voices().voices, key=lambda v: v.name)):
        voices[v.name] = "{}-{}-{}".format(i, v.name, v.ssml_gender.name)

    for voice in client.list_voices().voices:
        for code in voice.language_codes:
            if code not in voices_table:
                voices_table[code] = []
            voices_table[code].append(
                {'name': voice.name, 'display': voices[voice.name]})
    return voices_table


def to_audio(client, text, lang, rate=1.0, voice="en-US-Wavenet-D"):
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice_obj = texttospeech.VoiceSelectionParams(
        language_code=lang,
        name=voice,
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.OGG_OPUS,
        speaking_rate=rate,
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice_obj, audio_config=audio_config
    )

    # The response's audio_content is binary.
    buffer = BytesIO()
    buffer.write(response.audio_content)
    buffer.seek(0)
    seg = AudioSegment.from_file(buffer)
    return seg, seg.duration_seconds


def limit_texts(texts, limit=4500):
    new_texts = []
    for text in texts:
        text = text.strip()
        if len(text) == 0:
            continue
        if len(text) < limit:
            new_texts.append(text)
            continue
        for t in limit_texts(text.split('\n')):
            if len(t) < limit:
                new_texts.append(t)
                continue
            for tt in t.split('. '):
                new_texts.append(tt)
    return new_texts


def script_to_audio(client, src, lang='en-US', rate=0.9, voice="en-US-Wavenet-D"):
    if isinstance(src, str):
        if os.path.exists(src):
            thefile = open(src, 'r')
        else:
            thefile = StringIO(src)
    elif getattr(src, 'read', None) is not None:
        thefile = src
    raw = thefile.read().strip('---').strip()
    texts = limit_texts(raw.split('---'))
    audios = []
    for text in texts:
        a, seconds = to_audio(client, text, lang, rate=rate, voice=voice)
        audios.append(a)
        silent = pydub.AudioSegment.silent(duration=500)
        audios.append(silent)

    output = BytesIO()
    sum(audios).set_channels(2).export(output)
    return output
