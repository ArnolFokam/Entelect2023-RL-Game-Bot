FROM python:3.8

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY requirements.txt .
RUN python -m venv ecbot

RUN python --version
RUN pip --version

RUN ecbot/bin/pip --no-cache-dir install -r requirements.txt

COPY . .

ENTRYPOINT python -c 'print("Hello, World!")'