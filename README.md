**Note:** Do all your dev work on the train branch.

# Requires
- Python 3.8.16
- Docker
- Conda (Anaconda or Miniconda)

# How to run

## Environment
- Install dotnet following these [instructions](https://learn.microsoft.com/en-us/dotnet/core/install/linux-scripted-manual#scripted-install).
- Do not forget to set the dotnet command as an environment variable.
- Then run the following commands from the repository's directory
```bash
cd game/2023-CyFi
dotnet restore
dotnet publish --configuration Release --output ./publish
cd publish && dotnet CyFi.dll [port you want the server to run on]
```

## Agent
- Create python environment with `conda create -n ecbot python=3.8.16`
- Activate the environment with `conda activate ecbot`
- Install python things with `pip install -r requirements.txt`
- While env is running (on another window), run the following command:

```bash
python train.py --config-path=exps --config-name=ppo $(python new_hydra_dir_params.py) game_server_port=[port on which the server is running]
```

Note: if pygame crashes on some iris, swarst issues run the following command:

```bash
conda install -c conda-forge libstdcxx-ng
``` 
