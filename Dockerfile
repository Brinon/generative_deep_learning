FROM tensorflow/tensorflow:2.7.0-gpu


WORKDIR /opt

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8888
CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
