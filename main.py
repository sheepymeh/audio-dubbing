from asr import Whisper
from speaker_detection import SpeechBrain
from voice_generation import ElevenLabs
from io import BytesIO
from pydub import AudioSegment
from random import randrange

input_file = 'test.mp3'
output_file = 'test_o.mp3'

asr = Whisper('de', 'large-v3')
speaker_detection = SpeechBrain(thresh=.8)
voice_generation = ElevenLabs('<API_KEY>', voice_cloning_id=str(randrange(1_000_000, 10_000_000)))

audio = AudioSegment.from_file(input_file, frame_rate=16_000)
audio = audio.set_channels(1)
output_asr = asr(input_file)

if not output_asr:
	print('No audio detected')
	exit()

output_audio = AudioSegment.silent(duration=len(audio))

voices = []
for idx, segment in enumerate(output_asr):
	start, end = segment.start * 1000, segment.end * 1000
	with BytesIO() as f:
		audio[start:end].export(f, format='wav')
		voice = speaker_detection(f)
	segment.voice = voice
	if voice == len(voices):
		voices.append([idx])
	else:
		voices[voice].append(idx)

for voice in voices:
	files = []
	for idx in voice[:5]:
		start, end = output_asr[idx].start * 1000, output_asr[idx].end * 1000
		f = BytesIO()
		audio[start:end].export(f, format='mp3')
		files.append(f)
	voice_generation.new_voice(files)

for idx, segment in enumerate(output_asr):
	start, end = segment.start * 1000, segment.end * 1000
	output_segment = voice_generation(segment.voice, segment.text)
	output_segment = AudioSegment.from_file(BytesIO(output_segment))

	playback_speed = (len(output_segment) + 500) / (end - start)
	playback_speed = min(max(playback_speed, .8), 1.2)
	# output_segment = output_segment.speedup(playback_speed=playback_speed)

	output_audio = output_audio.overlay(output_segment, position=start)

output_audio.export(output_file)