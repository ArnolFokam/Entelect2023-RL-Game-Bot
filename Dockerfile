FROM python:3.8

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY online-bot-requirements.txt .
RUN python -m venv ecbot
RUN ecbot/bin/pip install signalrcore==0.9.5
RUN ecbot/bin/pip install gym==0.26.2

COPY . .

ENTRYPOINT python -c 'print("Hello, World!")'