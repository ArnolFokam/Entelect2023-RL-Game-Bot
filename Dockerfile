FROM python:3.8

ARG SOURCE_DIRECTORY

WORKDIR /app

COPY requirements.txt .
RUN python3 -m venv ecbot
RUN ls ecbot
RUN pip3 -r requirements.txt

COPY . .

ENTRYPOINT python3 -c 'print("Hello, World!")'