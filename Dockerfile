FROM python:3.8

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY requirements.txt .
RUN python -m venv ecbot

RUN ecbot/bin/pip install gym==0.26.2

RUN ecbot/bin/pip install -r online-bot-requirements.txt

COPY . .

ENTRYPOINT python -c 'print("Hello, World!")'