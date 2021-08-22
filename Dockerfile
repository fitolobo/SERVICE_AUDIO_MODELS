# Deep learning image
FROM pytorch/pytorch:latest

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev

RUN pip3 install scikit-learn
RUN pip3 install spicy
RUN pip3 install raiwidgets

RUN apt-get update
RUN apt-get install -y make
RUN pip3 install jupyterlab

RUN pip3 install librosa
RUN pip3 install matplotlib
RUN pip3 install uvicorn[standard]
RUN pip3 install fastapi
RUN pip3 install python-multipart
RUN pip3 install tensorflow
RUN pip3 install keras

CMD ["/bin/bash"]
