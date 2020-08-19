'''
Author: Tushar Goel

'''
from flask import Flask, render_template, session, request,Response,redirect
from flask_socketio import SocketIO, emit, disconnect
from werkzeug.utils import secure_filename
import scipy.io.wavfile
import numpy as np
from collections import OrderedDict
import sys,os
import numpy as np

import json
from datetime import datetime
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
from speakerDiarization import dia,label
from filter import main,t
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None
sns.set()

wav_data = bytearray()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, binary=True)
socketio.init_app(app, cors_allowed_origins="*")
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'mp4','mp3','wav','flac'}
#socketio = SocketIO(app, async_mode=async_mode)
#thread = None
#thread_lock = Lock()
diar = False
uploaded_file = ''
import array
import struct
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def float_to_16_bit_pcm(raw_floats):
	floats = array.array('f', raw_floats)
	samples = [int(sample * 32767) for sample in floats]
	raw_ints = struct.pack("<%dh" % len(samples), *samples)
	return raw_ints

@app.route('/')
def index():
	from filter import main
	session['audio'] = []
	file = open(os.path.join(app.config['UPLOAD_FOLDER'],'result.txt'),'w').close()
	t(0)
	return render_template('index.html')

@app.route('/analyze')
def analyze():
		plt.imread('static/index.png')
		figfile = BytesIO()
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		results = figdata_png.decode('utf-8')
		return render_template('charts.html',results=results,results1=results,results2=results)


@app.route('/visualize1')
def visualize1():
	try:
		df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],'result.txt'))
		df.columns = ['time','freq','category']
		lis = []
		for i in df['category'][:]:
			if i=='Positive':
				lis.append(1)
			if i=='Negative':
				lis.append(-1)
			if i=='Neutral':
				lis.append(0)

		df['scores'] = lis

		'''plt.plot(df[df['category']=='Positive']['time'],df[df['category']=='Positive']['freq'],'g',label='Positive')
								plt.plot(df[df['category']=='Negative']['time'],df[df['category']=='Negative']['freq'],'r',label = 'Negative')
								plt.plot(df[df['category']=='Neutral']['time'],df[df['category']=='Neutral']['freq'],'b',label='Neutral')
						'''

		x = df['time']
		y = df['freq']
		cmap = ListedColormap(['r', 'b', 'g'])
		norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
		points = np.array([x, y]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)

		lc = LineCollection(segments, cmap=cmap, norm=norm)
		lc.set_array(df['scores'])
		lc.set_linewidth(0.4)
		plt.gca().add_collection(lc)

		plt.xlim(x.min(), x.max())
		plt.ylim(y.min(),y.max())
		plt.title('Audio Analysis')
		plt.xlabel('TimeStamp (sec)')
		plt.gca().axes.yaxis.set_visible(False)
		plt.legend(loc='upper right')
		figfile = BytesIO()
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		results = figdata_png.decode('utf-8')
		plt.clf()
		return results
	except:
		plt.imread('static/index.png')
		figfile = BytesIO()
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		results = figdata_png.decode('utf-8')

		plt.clf()
		return results

@app.route('/visualize2')
def visualize2():
	try:
		df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],'result.txt'))
		df.columns = ['time','freq','category']
		lis = []
		for i in df['category'][:]:
			if i=='Positive':
				lis.append(1)
			if i=='Negative':
				lis.append(-1)
			if i=='Neutral':
				lis.append(0)

		df['scores'] = lis

		'''plt.plot(df[df['category']=='Positive']['time'],df[df['category']=='Positive']['freq'],'g',label='Positive')
								plt.plot(df[df['category']=='Negative']['time'],df[df['category']=='Negative']['freq'],'r',label = 'Negative')
								plt.plot(df[df['category']=='Neutral']['time'],df[df['category']=='Neutral']['freq'],'b',label='Neutral')
						'''

		s = ['Negative','Neutral','Positive']
		a = [(0+lis.count(-1))*100/len(lis),(0+lis.count(0))*100/len(lis),(0+lis.count(1))*100/len(lis)]
		plt.title('Analysis of Sentiment')
		plt.bar(s,a,color=['red','blue','green'])
		plt.ylabel('Percentage')
		figfile = BytesIO()
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		result1 = figdata_png.decode('utf-8')
		plt.clf()
		return result1
	except:
		plt.imread('static/index.png')
		figfile = BytesIO()
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		result1 = figdata_png.decode('utf-8')

		plt.clf()
		return result1

@app.route('/visualize3')
def visualize3():
	try:
		df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],'result.txt'))
		df.columns = ['time','freq','category']
		lis = []
		for i in df['category'][:]:
			if i=='Positive':
				lis.append(1)
			if i=='Negative':
				lis.append(-1)
			if i=='Neutral':
				lis.append(0)

		df['scores'] = lis

		'''plt.plot(df[df['category']=='Positive']['time'],df[df['category']=='Positive']['freq'],'g',label='Positive')
								plt.plot(df[df['category']=='Negative']['time'],df[df['category']=='Negative']['freq'],'r',label = 'Negative')
								plt.plot(df[df['category']=='Neutral']['time'],df[df['category']=='Neutral']['freq'],'b',label='Neutral')
						'''
		s = ['Negative','Neutral','Positive']
		a = [0+lis.count(-1),0+lis.count(0),0+lis.count(1)]
		plt.pie(a,labels=s,colors=['r','b','g'],labeldistance=None)
		plt.legend(loc='upper right')
		figfile = BytesIO()
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		result2 = figdata_png.decode('utf-8')

		plt.clf()
		return result2
	except:
		plt.imread('static/index.png')
		figfile = BytesIO()
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		result2 = figdata_png.decode('utf-8')

		plt.clf()
		return result2

c_t = 0
@app.route('/analyze-stop')
def stop(file = False):
	global c_t
	c_t = time.time()
	try:
		if file:
			df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],file))
		else:
			df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],'result.txt'))
		df.columns = ['time','freq','category']
		lis = []
		for i in df['category'][:]:
			if i=='Positive':
				lis.append(1)
			if i=='Negative':
				lis.append(-1)
			if i=='Neutral':
				lis.append(0)

		df['scores'] = lis

		'''plt.plot(df[df['category']=='Positive']['time'],df[df['category']=='Positive']['freq'],'g',label='Positive')
								plt.plot(df[df['category']=='Negative']['time'],df[df['category']=='Negative']['freq'],'r',label = 'Negative')
								plt.plot(df[df['category']=='Neutral']['time'],df[df['category']=='Neutral']['freq'],'b',label='Neutral')
						'''

		x = df['time']
		y = df['freq']
		cmap = ListedColormap(['r', 'b', 'g'])
		norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
		points = np.array([x, y]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)

		lc = LineCollection(segments, cmap=cmap, norm=norm)
		lc.set_array(df['scores'])
		lc.set_linewidth(0.4)
		plt.gca().add_collection(lc)
		plt.gca().legend(('Negative','Neutral','Positive'))
		plt.xlim(x.min(), x.max())
		plt.ylim(y.min(),y.max())
		plt.title('Audio Analysis')
		plt.xlabel('TimeStamp (sec) ')
		plt.gca().axes.yaxis.set_visible(False)
		plt.legend(loc='upper right')
		figfile = BytesIO()
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		result = figdata_png.decode('utf-8')
		plt.clf()
		
		s = ['Negative','Neutral','Positive']
		a = [(0+lis.count(-1))*100/len(lis),(0+lis.count(0))*100/len(lis),(0+lis.count(1))*100/len(lis)]
		plt.title('Analysis of Sentiment')
		plt.bar(s,a,color=['red','blue','green'])
		plt.ylabel('Percentage')
		plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'],'Sentiment_Bar_{}.png'.format(c_t)))
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		result1 = figdata_png.decode('utf-8')
		plt.clf()

		plt.pie(a,labels=s,colors=['r','b','g'],labeldistance=None)
		plt.legend(loc='best')
		plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'],'Sentiment_Pie_{}.png'.format(c_t)))
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		result2 = figdata_png.decode('utf-8')

		plt.clf()

		return render_template('analyze.html',results=result,results1=result1,results2=result2)

	except:
		return 'Please Record your Voice then after this page will be visible'

n_speakers = 0
def visualizer(file):

		df = pd.read_csv(file)
		df.columns = ['time','freq','category']
		lis = []
		for i in df['category'][:]:
			if i=='Positive':
				lis.append(1)
			if i=='Negative':
				lis.append(-1)
			if i=='Neutral':
				lis.append(0)

		df['scores'] = lis

		'''plt.plot(df[df['category']=='Positive']['time'],df[df['category']=='Positive']['freq'],'g',label='Positive')
								plt.plot(df[df['category']=='Negative']['time'],df[df['category']=='Negative']['freq'],'r',label = 'Negative')
								plt.plot(df[df['category']=='Neutral']['time'],df[df['category']=='Neutral']['freq'],'b',label='Neutral')
						'''

		x = df['time']
		y = df['freq']
		cmap = ListedColormap(['r', 'b', 'g'])
		norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
		points = np.array([x, y]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)

		lc = LineCollection(segments, cmap=cmap, norm=norm)
		lc.set_array(df['scores'])
		lc.set_linewidth(0.4)
		plt.gca().add_collection(lc)
		plt.gca().legend(('Negative','Neutral','Positive'))
		plt.xlim(x.min(), x.max())
		plt.ylim(y.min(),y.max())
		plt.gca().axes.yaxis.set_visible(False)
		plt.title('Audio Analysis')
		plt.xlabel('TimeStamp (sec) ')
		plt.legend(loc='upper right')
		figfile = BytesIO()
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		result = figdata_png.decode('utf-8')
		plt.clf()
		
		s = ['Negative','Neutral','Positive']
		a = [(0+lis.count(-1))*100/len(lis),(0+lis.count(0))*100/len(lis),(0+lis.count(1))*100/len(lis)]
		plt.title('Analysis of Sentiment')
		plt.bar(s,a,color=['red','blue','green'])
		plt.ylabel('Percentage')
		plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'],'Sentiment_Bar_{}.png'.format(c_t)))
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		result1 = figdata_png.decode('utf-8')
		plt.clf()

		plt.pie(a,labels=s,colors=['r','b','g'],labeldistance=None)
		plt.legend(loc='best')
		plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'],'Sentiment_Pie_{}.png'.format(c_t)))
		plt.savefig(figfile, format='png')
		figfile.seek(0)  # rewind to beginning of file
		figdata_png = figfile.getvalue()
		figdata_png = base64.b64encode(figdata_png)
		result2 = figdata_png.decode('utf-8')

		plt.clf()

		return result,result1,result2



@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
	global uploaded_file
	global n_speakers

	if request.method == 'POST':
		
		# check if the post request has the file part
		if 'file' not in request.files:
			
			return redirect(request.url)
		file = request.files['file']
		try:
			n_speakers = int(request.form['speaker'])
		except:
			n_speakers = 0
		print(n_speakers)
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			return redirect(request.url)
		if file and allowed_file(file.filename):
			global uploaded_file
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			uploaded_file =  os.path.join(app.config['UPLOAD_FOLDER'], filename)
			check_file = os.path.exists(uploaded_file)
		return render_template('index.html',url=True,upload_message = 'File Uploaded.Click Analyze',check_file=check_file)
	return render_template('index.html',url = True)

@app.route('/analyze_file',methods=['GET','POST'])
def analyze_file():
	global uploaded_file
	global n_speakers
	global diar
	global no_of_speaker
	global info
	diar = True
	
	if request.method == 'POST':
		try:
			speaker = int(request.form['speaker'])
		except:
			speaker = 0
		diar = False
		if speaker == 0:
			n_speakers = speaker
			try:
				result,result1,result2 = visualizer(os.path.join(app.config['UPLOAD_FOLDER'],'result.txt'))
				message = ''
			except:
				result = ''
				result1 = ''
				result2 = ''
				message = 'Wait for the Process to complete'
		else:
			try:
				n_speakers = speaker
				try:
					result,result1,result2 = visualizer(os.path.join(app.config['UPLOAD_FOLDER'],'result_{}.txt'.format(speaker-1)))
					message = ''
				except:
					result = ''
					result1 = ''
					result2 = ''
					message = 'Wait for the Process to complete.'
			except:
				result = ''
				result1 = ''
				result2 = ''
				message = 'Speaker Not Found. Please try with different speakers.'

		return render_template('speaker.html',results=result,results1=result1,results2=result2,speaker=speaker,message=message,info=info)
	
	return render_template('files.html',speaker = 0)

@app.route('/diarization')
def diarization():
	global diar
	global uploaded_file
	global n_speakers

	if diar:
			# print(diar,uploaded_file,n_speakers)
		uploaded_file_new_name = uploaded_file.split('.')[0]+'_dsw_formatted.wav'
		if not os.path.exists(uploaded_file_new_name):

			os.system(f'ffmpeg -i {uploaded_file} -ac 1 -ar 16000 {uploaded_file_new_name}')
			uploaded_file = uploaded_file_new_name

		else:
			print('File Already formatted')
			uploaded_file = uploaded_file_new_name
		diar = False
		
		
		return 'Diarization in Process'
	elif diar == False:
		return 'Diarization in Process'
	elif diar == None:
		return 'Enter the Speaker No'
no_of_speaker = 2
info = ''
@app.route('/message')
def mess():
	return 'Thanks for Uploading. File is in Process.'
@app.route('/dia')
def clustering():
	global diar
	global uploaded_file
	global n_spekaers
	global no_of_speaker
	global info
	if diar:
		uploaded_file_new_name = uploaded_file.split('.')[0]+'_dsw_formatted.wav'
		if not os.path.exists(uploaded_file_new_name):

			os.system(f'ffmpeg -i {uploaded_file} -ac 1 -ar 16000 {uploaded_file_new_name}')
			uploaded_file = uploaded_file_new_name

		else:
			print('File Already formatted')
			uploaded_file = uploaded_file_new_name
		dia(uploaded_file,n_speakers)
		no_of_speaker = label()
		a = len(no_of_speaker)
		s = ','.join(str(int(sp)+1) for sp in sorted(no_of_speaker))
		diar = False
		info  = f'{ a } speaker Identified. Type 0 for whole Audio Visualization or Choose Speaker no from {s} for Respective Speaker sentiment and Click on Show button'
		return f'{ a } speaker Identified. Type 0 for whole Audio Visualization or Choose Speaker no from {s} for Respective Speaker sentiment and Click on Show button'
	#result,result1,result2 = visualizer(os.path.join(app.config['UPLOAD_FOLDER'],'result.txt'))
'''
	'''

@socketio.on('my_event', namespace='/test')
def test_message(message):
	session['receive_count'] = session.get('receive_count', 0) + 1
	emit('my_response',
		 {'data': message['data'], 'count': session['receive_count']})



@socketio.on('disconnect_request', namespace='/test')
def disconnect_request():
	session['receive_count'] = session.get('receive_count', 0) + 1
	emit('my_response',
		 {'data': 'Disconnected!', 'count': session['receive_count']})
	disconnect()


#@socketio.on('my_ping', namespace='/test')
#def ping_pong():
#    emit('my_pong')


@socketio.on('connect', namespace='/test')
def test_connect():
	#global thread
	#with thread_lock:
	#    if thread is None:
	#        thread = socketio.start_background_task(target=background_thread)
	session['audio'] = []
	emit('my_response', {'data': 'Connected', 'count': 0})

@socketio.on('sample_rate', namespace='/test')
def handle_my_sample_rate(sampleRate):
	session['sample_rate'] = sampleRate
	# send some message to front
	session['receive_count'] = session.get('receive_count', 0) + 1
	emit('my_response', {'data': "sampleRate : %s" % sampleRate, 'count': session['receive_count'] })

@socketio.on('audio', namespace='/test')
def handle_my_custom_event(audio):
	#session['audio'] += audio
	#session['audio'] += audio.values()

	values = OrderedDict(sorted(audio.items(), key=lambda t:int(t[0]))).values()
	session['audio'] += values
	pcm_audio = float_to_16_bit_pcm(session['audio'])
	main(audio = pcm_audio,folder=app.config['UPLOAD_FOLDER'])

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
	#my_audio = np.array(session['audio'], np.float32)
	#scipy.io.wavfile.write('out.wav', 44100, my_audio.view('int16'))
	#print(my_audio.view('int16'))

	# https://stackoverflow.com/a/18644461/466693
	sample_rate = 16000
	my_audio = np.array(session['audio'], np.float32)

	sindata = np.sin(my_audio)
	scaled = np.round(32767*sindata)
	newdata = scaled.astype(np.int16)
	scipy.io.wavfile.write('out.wav', sample_rate, newdata)

	session['audio'] = []
	print('Client disconnected', request.sid)

if __name__ == '__main__':
	socketio.run(app,debug=True)