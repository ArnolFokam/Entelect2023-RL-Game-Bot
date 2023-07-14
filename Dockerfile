FROM public.ecr.aws/m5z5a5b2/languages/python_pytorch:2021

ARG ONLINE_MODEL_PATH

RUN echo "ONLINE_MODEL_PATH: $ONLINE_MODEL_PATH"
WORKDIR /app

RUN python -m venv ecbot
COPY online-bot-requirements.txt .
RUN ecbot/bin/pip --no-cache-dir install -r online-bot-requirements.txt

# TODO: copy the online bot code
COPY . .

ENTRYPOINT ecbot/bin/python -c 'print("Hello, World!")'