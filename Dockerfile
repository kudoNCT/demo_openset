FROM python:3.8.12-bullseye

WORKDIR /demo_openset-master
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install --upgrade pip
RUN pip install torch==1.8.0+cpu \
    torchvision==0.9.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -r requirements.txt
EXPOSE 5000
CMD gunicorn svr_model:app
