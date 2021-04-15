from w2l.utils.tts import create_client, list_voices, script_to_audio
import os


def test_to_audio():
    credential_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not os.path.exists(credential_path):
        print("bypass test, the credential file doesn't exist")
        return
    client = create_client()
    output = script_to_audio(client, "hello")
    open('tests/data/hello.wav', 'wb').write(output.read())
