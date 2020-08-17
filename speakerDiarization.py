from audioSegmentation import speaker_diarization
import pandas as pd
from pydub import AudioSegment
from filter import main_file,main
labels = [0,1]

def dia(filename,speakers,lda_dim = 0):
	global labels
	file = open('Speaker_Diarization.txt','w').close()
	main_file(filename,overall=True)
	timestamp,classes = speaker_diarization(filename=filename,n_speakers=speakers,lda_dim=lda_dim)
	file = open('Speaker_Diarization.txt','a')
	file.write('Timestamp,Classes\n')
	for i in range(len(timestamp)):
		file.write(f'{timestamp[i]},{int(classes[i])}\n')

	file.close()
	df = pd.read_csv('Speaker_Diarization.txt')
	labels = df['Classes'].unique()
	num_labels = len(labels)
	previous_label = df['Classes'][0]
	sda = dict()
	a = []
	segments = []
	for i in range(df.shape[0]):
		if i == 0:
			a.append(df['Classes'][i])
			a.append(df['Timestamp'][i])
		elif df['Classes'][i-1] == df['Classes'][i]:
			a.append(df['Timestamp'][i])

		else:
			segments.append([a[0],a[1],a[-1]])
			a = []
			a.append(df['Classes'][i])
			a.append(df['Timestamp'][i])
	p = []
	for label in labels:
		for segment in segments:
			if segment[0] == label:
				p.append([segment[1],segment[2]])

		sda[label] = p
		p = []

	speech = AudioSegment.from_wav(filename)
	for key,value in sda.items():
		speaker = 0
		for parts in value:
			speaker += speech[parts[0]*1000:parts[1]*1000]

		file = filename.split('.')[0]+str('_speaker_')+str(key)+'.wav'
		print(speaker)
		print(key,value)
		if len(value)==0:
			labels = list(labels)
			labels.remove(key)
		else:
			speaker.export(file,format='wav')
			print('Files Exported')
			main_file(file = file,folder='uploads',labels=labels)
	

def label():
	global labels
	return labels