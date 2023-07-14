FROM public.ecr.aws/m5z5a5b2/languages/python_pytorch:2021

ARG ONLINE_MODEL_PATH
ENV model_path=$ONLINE_MODEL_PATH

WORKDIR /app

RUN python -m venv ecbot-env
COPY requirements.txt .
RUN ecbot-env/bin/pip --no-cache-dir install -r requirements.txt

# copy the online bot code
COPY ecbot/ ecbot/ 

# copy trained bot weights
COPY ${ONLINE_MODEL_PATH}/ bot/

# copy the play script
COPY play_online.py .

RUN ls -la

ENTRYPOINT ecbot-env/bin/python play_online.py bot/