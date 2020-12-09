from pathlib import Path
import argparse
import subprocess
import os

ext = ".dat"


def main():

    parser = argparse.ArgumentParser(
        description="Save the extended metadata from the footer of a .dat file as json and xml "
        "files in the same directory as the .dat file. This script exists to "
        "work around a the fact that certain important metadata in .dat files is "
        " in an undocumented region in the end of the file. This script uses a "
        "LabVIEW executible written by Dan Milkie to serialize the metadata."
    )

    parser.add_argument(
        "-s",
        "--data_root",
        help="Path to a directory containing directories containing .dat files",
        required=True,
    )

    parser.add_argument(
        "-e",
        "--exe_path",
        help="Path to the executible which extracts the metadata from the .dat files",
        required=True,
    )

    args = parser.parse_args()
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(
            f"The data root path {args.data_root} could not be found."
        )
    elif not os.path.exists(args.exe_path):
        raise FileNotFoundError(f"The exe path {args.exe_path} could not be found.")
    toProcess = []
    rawDirs = Path(args.data_root).glob(f"*/raw/")
    toProcess.extend([str(list(sorted(x.glob(f"*{ext}")))[0]) for x in rawDirs])
    cmds = [[args.exe_path, t] for t in toProcess]
    [subprocess.run(c) for c in cmds]


if __name__ == "__main__":
    main()
