FROM public.ecr.aws/m5z5a5b2/languages/python_pytorch:2021

ARG SOURCE_DIRECTORY

WORKDIR /app

RUN python -m venv ecbot

# only woks when not in requirements.txt
RUN ecbot/bin/pip install signalrcore==0.9.5
RUN ecbot/bin/pip install gym==0.26.2

COPY . .

ENTRYPOINT python -c 'print("Hello, World!")'