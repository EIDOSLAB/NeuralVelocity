FROM python:3.8.8

RUN pip3 install --upgrade pip

RUN pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install scikit_learn==1.2.0 wandb==0.13.9

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Add the eidoslab group to the image
# not sure it is really needed but ok
RUN addgroup --gid 1337 eidoslab

RUN mkdir /.config
RUN chmod 775 /.config
RUN chown -R :1337 /.config

RUN mkdir /.cache
RUN chmod 775 /.cache
RUN chown -R :1337 /.cache

RUN mkdir /neural_velocity
RUN chmod 775 /neural_velocity
RUN chown -R :1337 /neural_velocity

RUN mkdir /neural_velocity/src
RUN chmod 775 /neural_velocity/src
RUN chown -R :1337 /neural_velocity/src

RUN mkdir /scratch
RUN chmod 775 /scratch
RUN chown -R :1337 /scratch

COPY requirements.txt /neural_velocity
COPY setup.py /neural_velocity

WORKDIR /neural_velocity/

RUN pip3 install -r requirements.txt

RUN pip3 install -e .

COPY src /neural_velocity/src

WORKDIR /neural_velocity/src/
