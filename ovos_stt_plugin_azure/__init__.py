import requests
from ovos_plugin_manager.stt import STT
from ovos_utils import classproperty
from ovos_plugin_manager.utils.audio import AudioData, AudioFile
from typing import Optional
from ovos_utils.log import LOG


class OVOSAzureSTT(STT):
    def __init__(self, config=None):
        super().__init__(config)
        self.lang_override = self.config.get("lang")
        self.key = self.config.get("key")
        self.region = self.config.get("region", "westeurope")
        self.profanity = self.config.get("profanity", "raw")

    def execute(self, audio: AudioData, language: Optional[str]=None):
        lang = self.lang_override or language or self.lang
        headers = {
            'Content-type': 'audio/wav;codec="audio/pcm";',
            'Ocp-Apim-Subscription-Key': self.key
        }
        url = f"https://{self.region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        raw_response = requests.request("POST", url, headers=headers,
                                    data=audio.get_wav_data(),
                                    params={"language": lang, "profanity": self.profanity})
        LOG.info(f"url: {url}, lang: {lang}, profanity: {self.profanity}, response: {raw_response.content}")
        response = raw_response.json()
        return response["DisplayText"]

    @classproperty
    def available_languages(cls) -> set:
        """Return languages supported by this TTS implementation in this state
        This property should be overridden by the derived class to advertise
        what languages that engine supports.
        Returns:
            set: supported languages
        """
        return set()  # TODO


if __name__ == "__main__":
    import os

    engine = OVOSAzureSTT({"key": 'XXX'})

    # inference
    jfk = f"{os.path.dirname(__file__)}/jfk.wav"
    with AudioFile(jfk) as source:
        audio = source.read()

    pred = engine.execute(audio, language="en-us")
    print(pred)
