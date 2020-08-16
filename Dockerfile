FROM continuumio/anaconda3:5.2.0
COPY . /voice
WORKDIR /voice
RUN conda install -c conda-forge/label/gcc7 hmmlearn
RUN pip install webrtcvad-wheels
RUN pip install -r requirements.txt
EXPOSE 5001 
ENTRYPOINT [ "python" ] 
CMD ["app.py"] 