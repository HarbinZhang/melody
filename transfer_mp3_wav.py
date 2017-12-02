from pydub import AudioSegment
filepath = './MarBlue.mp3'
output_filepath = './newone.wav'

sound = AudioSegment.from_mp3(filepath)
sound = sound.set_channels(1)
sound.export(output_filepath, format="wav")