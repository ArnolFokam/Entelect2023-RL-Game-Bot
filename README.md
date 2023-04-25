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


# TODO

## High Priority
- [ ] Code Input command class to test the agent **(features)**
- [ ] Build our first learning agent class ([DQN network](https://arxiv.org/pdf/1312.5602.pdf)) **(features)**
- [ ] Improve training pipeline **(features)**
    - [ ] Trainer class
    - [ ] Optimization
    - [ ] Logging (Wandb)
    - [ ] Hyper-parameter Tuning 
- [ ] Add endpoint on the game server to restart the game at the end of each episodes **(features)**
- [ ] Add distance from collectables and enemies as agent state **(features)**
- [ ] Build a* enemies (CAUTION: limit the diversity of oponents) **(features)** best steal, best collector, best escaper

## Low Priority

- [ ] Build the inference class that plays the game using a trained agent **(features)**
- [ ] Add support for distributed training to train agents in parallel **(features)**
- [ ] Check bugs on server and submit it to [entellect repo](https://github.com/EntelectChallenge/2023-Cy-Fi) **(features)**

- [ ] Add self-play mechanism **(features)**