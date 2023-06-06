# Requires
- Python 3.8.16
- Docker
- Conda (Anaconda or Miniconda)

# How to run

## Environment
- Build docker image and run environment with `cd game && ./run.sh`
- Takes roughly 10 to 20 mins

## Agent
- Create python environment with `conda create -n ecbot python=3.8.16`
- Activate the environment with `conda activate ecbot`
- Install python things with `pip install -r requirements.txt`
- While env is running, run `python -m ecbot.train` on another terminal tab
- `conda install -c conda-forge libstdcxx-ng` if pygame crashes on some iris, swarst issues

## Modifications made to the environment
- The seed of the environment is random
    - 
- The agent is spawned to random location between the four available locations
    - 