---


---

<h1 id="voice-sentiment-analyzer">VOICE SENTIMENT ANALYZER</h1>
<p><b>Voice Sentiment Analyzer</b> is a Model created for the purposes of analysis of real time audio or recorded audio for many <strong>service based companies</strong>, which give insights of <strong>feedback</strong> and <strong>queries</strong> along with <strong>sentiment</strong> directly to them without any manual work on each call received from the customers. This help many organisations to improve their service providing systems <strong>efficiently</strong> and <strong>effectively</strong>.</p>
<h2 id="contents">CONTENTS</h2>
<ul>
<li><a href="#architecture">Architecture</a></li>
<li><a href="#research">Research</a></li>
<li><a href="#usage">Usage</a>
<ul>
<li><a href="#recording">Recording</a></li>
<li><a href="#uploading">Uploading File</a></li>
</ul>
</li>
<li><a href="#deployment">Deployment</a></li>
<li><a href="#furtherenhancement">Further Enhancement</a></li>
</ul>
 <h2 id="architecture">ARCHITECTURE</h2>
<p><img src="https://drive.google.com/uc?export=view&amp;id=1aR7EUXVlIHbIAfd-8C9mBJfS536DrYAs" alt="enter image description here"><br>
Above Image is the Voice Analysis Pipeline. Let’s Understand every phase of it’s working:</p>
<ol>
<li><u><strong>Data Collection</strong></u>: Using the Flask API, we will collect the audio in the form of Real time Audio or Recording through Mic or By Uploading the File to the server or by Connecting Institutional or Organisation Database with it.<br></li>
<li><u><strong>Preprocessing</strong></u>: Preprocessing of Audio require when audio is being uploaded or database connect to them. Preprocessing includes <strong>Speaker Diarization</strong> Process which through some clustering mechanisms separate out different speaker voices into different audio files. After Diarization process, there will be <strong>Voice Activity Detection</strong> , which separate <strong>speech</strong> and <strong>silence</strong> into different audio chunks for fast working of audio analysis.<br></li>
<li><u><strong>Model Building</strong></u>: Model Building Process in our Product makes it unique among its competitors, Our Model comprises of two Models:<br>
<p></p><ul><li><u><b>Speech to Text and Analysis</b></u></li>It converts incoming speech parts into Text using Google Speech-to-Text API. After conversion, sentiment of text is obtained from pretrained model and will be classified into different sentiment as <strong>Positive</strong>, <strong>Neutral</strong> and <strong>Negative</strong>. <br><br>
<li><u><b>Speech Emotion Analyzer</b></u></li>Using Pretrained Emotion Analyzer, we classify emotion of audio broadly into three categories as: **Happy, Neutral ** and <strong>Angry</strong> emotions .</ul><p></p><br></li>
<li><u><strong>Result Generation</strong></u>: We will combine result obtained from two models into one to get our final result. We will visualize these results into various Graphs as Shown below:<br>
<img src="https://drive.google.com/uc?export=view&amp;id=1RJ8DUuE_JDRov05Q37k0HvlgyxoHCo37" alt="Result Visualization"></li>
</ol>
<h2 id="research">RESEARCH</h2>

