FROM public.ecr.aws/lambda/python:3.11-preview

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./ecbot ./ecbot

ENTRYPOINT python3 -m ecbot.train 