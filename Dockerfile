FROM amazon/aws-lambda-python:3.8.2023.04.18.00

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./ecbot ./ecbot

ENTRYPOINT python3 -m ecbot.train 