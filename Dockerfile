FROM python:3.8

ARG SOURCE_DIRECTORY

WORKDIR /app

RUN python -m venv ecbot

# only woks when not in requirements.txt
RUN ecbot/bin/pip install signalrcore==0.9.5
RUN ecbot/bin/pip install gym==0.26.2

COPY . .

ENTRYPOINT python -c 'print("Hello, World!")'