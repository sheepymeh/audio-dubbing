import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from io import BytesIO

class SpeechBrain:
	def __init__(self, thresh: float = .8, model: str = 'speechbrain/spkrec-ecapa-voxceleb'):
		self.thresh = thresh
		self.model = EncoderClassifier.from_hparams(source=model)
		self.voice_embeddings = None
		self.voice_embedding_quality = []

	def __call__(self, file: BytesIO) -> int:
		signal, fs = torchaudio.load(file) # TODO: convert to BytesIO instead or use pydub
		embeddings = self.model.encode_batch(signal).unsqueeze(0)
		duration = len(signal[0]) / fs

		if self.voice_embeddings is None:
			self.voice_embeddings = embeddings
			self.voice_embedding_quality.append(duration)
			return 0

		similarities = torch.nn.functional.cosine_similarity(embeddings, self.voice_embeddings, dim=-1).flatten(start_dim=1)
		similary_max, matched_voice = [x.item() for x in similarities.max(0, True)]
		if similary_max < self.thresh:
			# New voice detected
			self.voice_embeddings = torch.cat((self.voice_embeddings, embeddings))
			self.voice_embedding_quality.append(duration)
			matched_voice = len(self.voice_embeddings) - 1
		elif self.voice_embedding_quality[matched_voice] < duration:
			self.voice_embeddings[matched_voice] = embeddings
			self.voice_embedding_quality[matched_voice] = duration
		return matched_voice