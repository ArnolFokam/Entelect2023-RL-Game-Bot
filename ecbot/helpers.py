from io import BytesIO
import os
import re
import string
import random
import datetime
import numpy as np
from typing import Any, Iterable, List

def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)

def get_chunks(data: List[Any], chunck_num: int) -> Iterable[List[Any]]:
    """
    Divide list of elements into chuncks of `chunk_num` elements each
    except the last chunk if the the total number of elements is not
    divisible by `chunk_num`
    Args:
        data (List[Any]): list of elements
        chunck_num (int): number of elements each chuck should contain
    Returns:
        Iterable[List[Any]]: generator chunks
    """
    for i in range(0, len(data), chunck_num):
        yield data[i : i + chunck_num]


def get_dir(*paths) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name
    Args:
        paths (List[str]): list of string that constitutes the path
    Returns:
        str: the created or existing path
    """
    directory = os.path.join(*paths)
    os.makedirs(directory, exist_ok=True)
    return directory

def generate_random_string(length: int = 10):
    """
    generate random alphanumeric characters

    Args:
        legnth (int): length of the string to be generated
    """
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def get_current_date() -> str:
    """
    returns the current date in format => 20XX-12-30
    """
    return datetime.datetime.now().strftime("%Y-%m-%d")


def get_current_time() -> str:
    """
    returns the current time in format => 24-60-60
    """
    return datetime.datetime.now().strftime("%H-%M-%S")


def get_new_run_dir_params():
    # hydra uses timestamp to create logs and
    # stuffs. This might be a problem when running
    # simultaneous process. this ensures that no
    # hydra runs share the same folders
    d_suffix = os.path.join(
        get_current_date(), get_current_time(), generate_random_string()
    )
    run_dir = os.path.join("results", d_suffix)
    multirun_dir = os.path.join("results/multirun", d_suffix)
    return {"hydra.run.dir": run_dir, "hydra.sweep.dir": multirun_dir}


def is_path_creatable(dir: str):
    """check is the directory is creatable

    Args:
        dir (str): directory

    Returns:
        bool: check result
    """
    valid_dir_pattern = re.compile(
        "^((\.|\.\.)(\/)?)?((.+)\/([^\/]+))?(\/)?(.+)?$"  # noqa: W605 escapes are fine flake!
    )
    has_no_space = " " not in dir
    not_parent_of_cwd = (
        not dir.startswith("/") and not dir.startswith("..")
    ) or dir.startswith(os.getcwd())
    return has_no_space and bool(re.match(valid_dir_pattern, dir)) and not_parent_of_cwd


def has_valid_hydra_dir_params(arguments: List[str]):
    """
    check list of arguments has hydra run dir

    Args:
        arguments (List[str]): list of arguments
    """

    def is_valid_param(argument: str, param: str):
        """
        check the argument is in the forman "{param}={valid path}"

        Args:
            argument (str): potential argument
            param (str): param key

        Returns:
            bool: result of the check
        """
        pair = argument.split("=")
        return len(pair) == 2 and pair[0] == param and is_path_creatable(pair[1])

    has_hyda_run_dir = any([is_valid_param(arg, "hydra.run.dir") for arg in arguments])
    has_hydra_sweep_dir_param = any(
        [is_valid_param(arg, "hydra.sweep.dir") for arg in arguments]
    )

    return has_hyda_run_dir and has_hydra_sweep_dir_param