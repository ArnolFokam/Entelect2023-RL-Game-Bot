FROM public.ecr.aws/m5z5a5b2/languages/python_pytorch:2021

ARG SOURCE_DIRECTORY

WORKDIR /app

RUN python -m venv ecbot
COPY online-requirements.txt .
RUN ecbot/bin/pip --no-cache-dir install -r online-requirements.txt

# TODO: copy the online bot code
COPY . .

ENTRYPOINT ecbot/bin/python -c 'print("Hello, World!")'