FROM python:3.8

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY online-bot-requirements.txt .
RUN python -m venv ecbot
RUN ecbot/bin/pip install -r online-bot-requirements.txt

COPY . .

ENTRYPOINT python -c 'print("Hello, World!")'