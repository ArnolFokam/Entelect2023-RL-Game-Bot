FROM amazon/aws-lambda-python:3.11.2023.07.13.17

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./ecbot ./ecbot

ENTRYPOINT python3 train.py --config-path=exps --config-name=ppo $(python new_hydra_dir_params.py)