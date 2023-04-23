# From Manuel
- Create python environment
- Install python things with pip install -r requirements.txt
- Build docker image and run environment with `cd environment && ./run.sh`
- Takes roughly 10 to 20 mins
- While env is running, run `python -m ecbot.main`

# TODO

## High Priority
- [ ] Code Input command class to test the agent (features)
- [ ] Build our first learning agent class ([DQN network](https://arxiv.org/pdf/1312.5602.pdf)) (features)
- [ ] Improve training pipeline (features)
    - [ ] Trainer class
    - [ ] Optimization
    - [ ] Logging (Wandb)
    - [ ] Hyper-parameter Tuning 
- [ ] Add endpoint on the game server to restart the game at the end of each episodes

## Low Priority

- [ ] Build the inference class that plays the game using a trained agent (features)
- [ ] Add support for distributed training to train agents in parallel (features)