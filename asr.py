# from faster_whisper import WhisperModel
import whisper
from dataclasses import dataclass

@dataclass
class ASRSegment:
	start: float
	end: float
	text: str
	voice: int = None

class Whisper:
	def __init__(self, target_lang: str, model: str = 'base', no_speech_thresh: float = .7):
		# self.model = WhisperModel(model, device='cuda', compute_type='int8')
		self.model = whisper.load_model(model)
		self.target_lang = target_lang
		self.no_speech_thresh = no_speech_thresh

	def __call__(self, filename: str) -> list[ASRSegment]:
		result = self.model.transcribe(filename, task='translate', language=self.target_lang) # TODO: can this run on PyDub input instead?
		segments = [ASRSegment(s['start'], s['end'], s['text']) for s in result['segments'] if s['no_speech_prob'] < self.no_speech_thresh]
		return segments