# VOICE SENTIMENT ANALYZER
**Voice Sentiment Analyzer** is a Model created for the purposes of analysis of real time audio or recorded audio for many **service based companies** which give insights of **feedback** and **queries** along with **sentiment** directly to them without any manual work on each call received from the customers. This help many organisations to improve their service providing systems **efficiently** and **effectively**.
## CONTENTS

- <a href="#architecture">Architecture</a>
 - <a href="#research">Research</a>
 - <a href="#usage">Usage</a>
		 - <a href="#recording">Recording</a>
		  - <a href="#uploading">Uploading File</a>
- <a href="#deployment">Deployment</a>
- <a href="#furtherenhancement">Further Enhancement</a>

 <h2 id="architecture">ARCHITECTURE</h2>
 <img src='https://drive.google.com/uc?export=view&amp;id=1aR7EUXVlIHbIAfd-8C9mBJfS536DrYAs'>
Above Image is the Voice Analysis Pipeline. Let’s Understand every phase of it’s working:

1. <u>**Data Collection**</u>: Using the Flask API, we will collect the audio in the form of Real time Audio or Recording through Mic or By Uploading the File to the server or by Connecting Institutional or Organisation Database with it.<br></li>
2. <u>**Preprocessing**</u>: Preprocessing of Audio require when audio is being uploaded or database connect to them. Preprocessing includes  **Speaker Diarization** Process which through some clustering mechanisms separate out different speaker voices into different audio files. After Diarization process, there will be **Voice Activity Detection**, which separate **speech** and **silence**  into different audio chunks for fast working of audio analysis.<br>
3. <u>**Model Building**</u>: Model Building Process in our Product makes it unique among its competitors, Our Model comprises of two Models:<br><br>
	 - <u>**Speech to Text and Analysis**</u>
 It converts incoming speech parts into Text using Google Speech-to-Text API. After conversion, sentiment of text is obtained from pretrained model and will be classified into different sentiment as **Positive**, **Neutral** and **Negative** .<br><br>
	- <u>**Speech Emotion Analyzer**</u>
Using Pretrained Emotion Analyzer, we classify emotion of audio broadly into three categories as: **Happy, Neutral ** and <strong>Angry</strong> emotions .<br><br>
4. <u>**Result Generation**</u>: We will combine result obtained from two models into one to get our final result. We will visualize these results into various Graphs as Shown below:<br>
<img src="https://drive.google.com/uc?export=view&amp;id=1RJ8DUuE_JDRov05Q37k0HvlgyxoHCo37" alt="Result Visualization">

<h2 id="research">RESEARCH</h2>
Now we are going to discuss how we reach to our final approach of selection of Model and creation of pipeline along with problems faced and their respective solutions.

1. <u> **Speaker Diarization** </u>: Speaker Diarization is the process in which we apply different clustering techniques to the features of the audio to seperate speakers present, to have a detailed indivisual sentiment also with overall audio sentiment.
	<p> Our first approach goes to use Google supervised diarization algorithm for clustering mechanism known as <b>UIS-RNN</b> ( Unbounded Interleaved-State Recurrent Neural Network ) and combine with VGG-16 voice feature extraction but having a less accuracy having a overlapping problem. </p>
	<p> Currently we are using  <b>PyAudioAnalysis </b> which uses <b>K-Means </b> and <b> SVM</b> to segregate speaker voices from audio file where number of  speakers may be defined by user or it uses elbow method to determine appropriate number of clusters.</p>
	<p>In our further enhancements, we  will be using <b>RPNSD</b> (Regional Proposed Network Speaker Diarization) , most accurate and resolved overlapped problem.</p>

2. <u>**Voice Activity Detection** </u>: It is a technique which classify different parts of audio file into **speech** or **silence** categories. This helps audio into different speech chunks after removal of silence from audio which helps effecient working of Model.
		<p> We  have used **WebRTC Voice Activity Detection** algorithm  to classify the segments of audio into speech and silence .</p>

3. <u>**Speech Recognition**</u>: It is the process through which we convert speech into text in specific Languages: Currently we are converting them into 'English-Indian' and 'English-US'.

	<p>We first approach to <b>Mozilla's Open Source DeepSpeech Recognition Model</b> trained on American English and having a WER ( Word Error Rate ) of 5.83%. But due to its memory effeciency is low as it's file size is about 1.2 Gb , making memory insufficient on Cloud services.</p>
	<p> Then our Final Approach goes to <b>Google Speech to Text API</b> trained and provide Services on vast group of Languages with WER ( Word Error Rate ) of about 3.44%.It is Highly effecient in terms of size and performance</p>

4. <u><b>Emotion Recognition</b></u>: We have a objective to add this to our model for increasng accuracy, because the tone with text decides the overall sentiment, as we have done with our speech recognition part, we now move on to emotion recognition part. 

	<p> We have trained our own model on Various datasets available for Public use and got an accuracy of 88.14% which classify the audio files into 3 categories: Angry, Neutral and Happy.

5. <u>**Text Sentiment Analysis**</u>: Text sentiment is the process in which the analysis of the text is done whether the text has Positive , Negative or Neutral impact.

	<p>Our first approach is <b>VaderSentiment</b>, which uses <b>Dictionary Based Approach</b> for sentiment. In a simplicity manner, we can understand it as like it has a million of words in its dictionary and on the basis of that it decides the sentiment of it.</p>

	<p>Then we move on to another advanced approach: <b>Embedding Based Approach</b>,
	means that a meaning of word changes  with the  context in which it is used. It depends on the neighbourhood words , for which we use Embeddings, which means an N-Dimensional space vector where words are present whose similarities between each word derives from the Cosine Similarity or Euclidean Distance . Models which are based on this approach are <b>Flair</b> and <b> FastText </b>( by FaceBook ). </p>

	<p>This type of systems are mostly organisation based , means in some orgaisation consider  some kind of words to be negative like in Banks such as Fraud. So here we need combination of Both <b> Dictionary and Embedding Based Approach .</b>


<h2 id='usage'>USAGE</h2>
