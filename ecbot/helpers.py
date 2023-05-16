import os
import string
import random

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