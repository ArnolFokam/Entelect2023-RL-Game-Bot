FROM amazon/aws-lambda-python:3.11.2023.07.13.17

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY requirements.txt .

RUN python3 -m venv /opt/ecbot
RUN /opt/ecbot/pip -r requirements.txt

COPY . .

ENTRYPOINT python3 -c 'print("Hello, World!")'