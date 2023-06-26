import os
import random
import itertools
import subprocess
from typing import Dict, List, Union

import fire
from dotenv import load_dotenv
from omegaconf import ListConfig, OmegaConf

from ecbot.helpers import generate_random_string, get_chunks, get_dir
from new_hydra_dir_params import main as get_new_hypdra_dir_params

# TODO: the exclude is only practical for cases where the runs you want to exclude are not that big

load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")
CONDA_ENV_NAME = os.getenv("CONDA_ENV_NAME")
CONDA_HOME = os.getenv("CONDA_HOME")
LOGS_DIR = os.getenv("LOGS_DIR")
SCRIPTS_DIR = os.getenv("SCRIPTS_DIR")

LOCAL = "local"
SLURM = "slurm"
CHPC = "chpc"


def is_subset_dict(subset_dict: Dict, superset_dict: Dict):
    return subset_dict.items() <= superset_dict.items()

# slurm
# python create_runs.py --experiment decoupled --yaml_sweep dcc --max_runs_per_scripts 20 --max_parallel_runs 2 --computer slurm --exclude $(cat bad_ones_private) --partition_name bigbatch

# local 
# python create_runs.py --experiment decoupled --yaml_sweep dcc --max_runs_per_scripts 1 --max_parallel_runs 1 --computer local

# chpc
# python create_runs.py --experiment baseline --yaml_sweep bc --max_runs_per_scripts 1 --max_parallel_runs 1 --computer chpc --partition_name gpu_1 --ressources select=1:ncpus=10:ngpus=1 --chpc_project PROJ102

def main(
    experiment: str,
    yaml_sweep: str,
    computer: str,
    exclude: Union[str, List[str]] = None,
    partition_name: str = "bigbatch",
    ressources: str = "select=1:ncpus=10:ngpus=1",
    max_runs_per_scripts: int = 1,
    max_parallel_runs: int = 1,
    chpc_project: str = "PROJ123",
    # use_distributed: bool = False,
    run_immediately: bool = False,
):
    """
    Creates run scripts for HPC
    Args:
        experiment (str):  experiment to run
        yaml_sweep_file (str): file path containing the parameter to sweep through
        computer (str): computing devices on which this script we be run
        exclude (Union[str, List[str]], optional):  nodes to exclude. Defaults to None.
        partition_name (str, optional): Partition to run the code on. Defaults to "stampede".
        ressources (_type_, optional): ressources to use for PBS. Defaults to "select=1:ncpus=10:ngpus=1".
        max_runs_per_scripts (int, optional): maximum number of python call torun an experiment bet bash file ran. Defaults to 1.
        chpc_project (str, optional): project to use for PBS. Defaults to "PROJ123".
        use_distributed (bool, optional): should we run the code in distributed mode?
        run_immediately (bool, optional): should we run the scipts immediately? Defaults to False.
    """

    # create arrays of arguments command for the sweep
    suffix = "_private"
    sweep = OmegaConf.load(f"{ROOT_DIR}/exps/sweeps/{yaml_sweep}.yaml")
    excluded_sweeps = sweep.pop("excludes", []) or []
    keys, values = zip(*sweep.items())
    additional_sweep_arguments = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    # filter sweep
    additional_sweep_arguments = [
        args
        for args in additional_sweep_arguments
        if not any(
            [is_subset_dict(ex, args) for ex in excluded_sweeps]
        )  # is not in excluded sweeps
    ]
    random.shuffle(additional_sweep_arguments)

    # generate a command line run for each combination of hyperparameters obtained from the sweep yaml file
    
    counter = 0
    
    commands = []
    for arguments_chunk in get_chunks(additional_sweep_arguments, max_runs_per_scripts):
        command = ""
        
        for arguments in arguments_chunk:
            # {'torchrun  --standalone --nproc_per_node=gpu' if use_distributed else 'python'}
            run = f"python train.py --config-path=exps --config-name={experiment}  {get_new_hypdra_dir_params()}"
            for key, value in arguments.items():
                run += f" {key}="
                if isinstance(value, (list, ListConfig)):
                    # we need a special care for list values as hydra
                    # will throw an error if there are quotes in the list items
                    run += "["
                    for idx in range(len(value)):
                        if idx == 0:
                            run += f"{value[idx]}"
                        else:
                            run += f",{value[idx]}"
                    run += "]"
                else:
                    run += f"{value}"
            command += f"\n{run}"
            
            if computer != LOCAL and max_parallel_runs > 1:
                command += "&"
                
                # update command counter
                counter += 1
                
                if counter == max_parallel_runs:
                    command += "\nwait;"
                    counter = 0
                
        if computer != LOCAL and counter != 0:
            command += "\nwait;"
            
        commands.append(command)

    # Create Slurm File
    def get_bash_text(bsh_cmd):
        return f"""#!/bin/bash
{f"#SBATCH -p {partition_name}" if  computer == SLURM else ""}
{f"#SBATCH -N 1" if  computer == SLURM else ""}
{f"#SBATCH -t 72:00:00" if  computer == SLURM else ""}
{f"#SBATCH -x {exclude if isinstance(exclude, str) else ','.join(exclude)}" if isinstance(exclude, (str, list, tuple)) else ""}
{f"#SBATCH -J {experiment}" if  computer == SLURM else ""}
{f"#SBATCH -o {get_dir(f'{ROOT_DIR}/{LOGS_DIR}', 'outputs')}/{experiment}.%N.%j.out" if  computer == SLURM else ""}
{f"#SBATCH -e {get_dir(f'{ROOT_DIR}/{LOGS_DIR}', 'errors')}/{experiment}.%N.%j.err" if  computer == SLURM else ""}
{f"#PBS -N {experiment}" if  computer == CHPC else ""}
{f"#PBS -q {partition_name}" if  computer == CHPC else ""}
{f"#PBS -l {ressources}" if  computer == CHPC else ""}
{f"#PBS -P {chpc_project}" if  computer == CHPC else ""}
{f"#PBS -l walltime=12:00:00" if  computer == CHPC else ""}
{f"#PBS -o {get_dir(f'{ROOT_DIR}/{LOGS_DIR}', 'outputs')}/{experiment}.%q.%P.out" if  computer == CHPC else ""}
{f"#PBS -e {get_dir(f'{ROOT_DIR}/{LOGS_DIR}', 'errors')}/{experiment}.%q.%P.err" if  computer == CHPC else ""}
cd {ROOT_DIR}
{f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {CONDA_ENV_NAME}" if  computer == SLURM else ""}
{f"source /apps/chpc/chem/anaconda3-2021.11/etc/profile.d/conda.sh" if  computer == CHPC else ""}
{f"conda activate {CONDA_HOME}/envs/{CONDA_ENV_NAME}" if  computer == CHPC or computer == LOCAL else ""}
{bsh_cmd}
{"conda deactivate"}
"""
# {f"#PBS -M fokammanuel1@students.wits.ac.za" if  computer == CHPC else ""}
# {f"#PBS -m abe" if  computer == CHPC else ""}
# {f"export PATH=$PATH:/usr/local/cuda-11.8/bin"if  computer == SLURM else ""}
# {f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64"if  computer == SLURM else ""}

    directory = get_dir(f"{ROOT_DIR}/{SCRIPTS_DIR}")

    for cmd in commands:
        idx = generate_random_string()
        fpath = os.path.join(directory, f"{experiment}_{idx}{suffix}.bash")
        with open(fpath, "w+") as f:
            f.write(get_bash_text(cmd))

        # Run it
        if run_immediately:
            if computer == SLURM:
                ans = subprocess.call(f"sbatch {fpath}".split(" "))
            elif computer == CHPC:
                ans = subprocess.call(f"qsub {fpath}".split(" "))
            else:
                ans = subprocess.call(
                    f"""
                source {CONDA_HOME}/etc/profile.d/conda.sh
                conda activate {CONDA_ENV_NAME}
                bash {fpath}
                """,
                    shell=True,
                    executable="/bin/bash",
                )
            assert ans == 0
            print(f"Successfully called {fpath}")


if __name__ == "__main__":
    fire.Fire(main)