FROM amazon/aws-lambda-python:3.11.2023.07.13.17

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY requirements.txt requirements.txt

RUN python3 -m venv ecbot
RUN source ecbot/bin/activate
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

COPY ./ecbot ./ecbot

ENTRYPOINT python -m ecbot.train 