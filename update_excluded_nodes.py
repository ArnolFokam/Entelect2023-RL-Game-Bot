#!/usr/bin/python3

import glob
import os

if __name__ == "__main__":
    for filename in glob.glob("waiting_private/*.bash"):

        lines = ""

        # create update bash file
        with open(filename) as f:
            for line in f:
                if line.startswith("#SBATCH -x "):
                    prefix, option, nodes = line.split(" ")

                    if os.path.isfile("bad_ones_private"):
                        with open("bad_ones_private", "r") as bf:
                            # only read the first line
                            nodes = bf.readlines()[0] + "\n"
                        bf.close()

                    line = " ".join([prefix, option, nodes])

                lines += line

        f.close()

        # actually write the file
        with open(filename, "w") as f:
            f.write(lines)
        f.close()