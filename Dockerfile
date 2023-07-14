FROM public.ecr.aws/m5z5a5b2/languages/python:2021

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY requirements.txt .
RUN python3 -m venv ecbot
RUN ecbot/bin/pip3 --no-cache-dir install --upgrade pip
RUN ecbot/bin/pip3 --no-cache-dir install -r requirements.txt

COPY . .

ENTRYPOINT python3 -c 'print("Hello, World!")'