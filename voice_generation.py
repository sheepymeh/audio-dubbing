from elevenlabs import generate, play, set_api_key
from elevenlabs.api import Voice
from elevenlabs.api.base import API
from io import BytesIO
from random import randrange

class VoiceClone(API):
	name: str
	files_tuple: tuple
	description: str = ''
	labels: dict[str, str] = None
	api_key: str

class ElevenLabs:
	def __init__(self, api_key: str, voice_cloning_id: str):
		self.voices = []
		self.api_key = api_key
		self.voice_cloning_id = voice_cloning_id
		set_api_key(api_key)

	def new_voice(self, files: tuple[BytesIO]) -> None:
		self.voices.append(Voice.from_clone(VoiceClone(
			name=self.voice_cloning_id + str(len(self.voices)),
			files_tuple=[('files', (str(randrange(100_000_000, 1_000_000_000)), file, 'audio/mpeg')) for file in files],
			api_key=self.api_key
		)))

	def __call__(self, voice: int, text: str) -> bytes: # TODO: convert to BytesIO instead or use pydub
		output = generate(text=text, voice=self.voices[voice], api_key=self.api_key)
		# output = generate(text=text, voice='Bella', api_key=self.api_key)

		return output