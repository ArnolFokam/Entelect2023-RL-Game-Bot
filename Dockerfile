FROM public.ecr.aws/m5z5a5b2/languages/python:2021

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./ecbot ./ecbot

ENTRYPOINT python3 -m ecbot.main