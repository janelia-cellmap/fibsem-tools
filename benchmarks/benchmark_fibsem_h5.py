from fibsem_tools.io.fibsem_h5 import (
    create_aggregate_fibsem_h5_file_from_dat_filenames,
    load_fibsem_from_h5_file,
)
import h5py
from timeit import timeit
from datetime import datetime
import hdf5plugin
import os.path
import pathlib
import json


def get_dat_file_groups():
    # Need files from //dm11.hhmi.org/public/for_mark_k/Z0720-07m_BR_Sec18
    dat_files_125416 = [
        "Z0720-07m_BR_Sec18/Merlin-6257_21-05-20_125416_0-0-0.dat",
        "Z0720-07m_BR_Sec18/Merlin-6257_21-05-20_125416_0-0-1.dat",
        "Z0720-07m_BR_Sec18/Merlin-6257_21-05-20_125416_0-0-2.dat",
        "Z0720-07m_BR_Sec18/Merlin-6257_21-05-20_125416_0-0-3.dat",
    ]
    dat_files_125527 = [
        "Z0720-07m_BR_Sec18/Merlin-6257_21-05-20_125527_0-0-0.dat",
        "Z0720-07m_BR_Sec18/Merlin-6257_21-05-20_125527_0-0-1.dat",
        "Z0720-07m_BR_Sec18/Merlin-6257_21-05-20_125527_0-0-2.dat",
        "Z0720-07m_BR_Sec18/Merlin-6257_21-05-20_125527_0-0-3.dat",
    ]
    dat_files_203058 = [
        "Z0720-07m_BR_Sec22/Merlin-6281_21-06-18_203058_0-0-0.dat",
        "Z0720-07m_BR_Sec22/Merlin-6281_21-06-18_203058_0-0-1.dat",
    ]
    dat_files_203155 = [
        "Z0720-07m_BR_Sec22/Merlin-6281_21-06-18_203155_0-0-0.dat",
        "Z0720-07m_BR_Sec22/Merlin-6281_21-06-18_203155_0-0-1.dat",
    ]
    dat_files_203252 = [
        "Z0720-07m_BR_Sec22/Merlin-6281_21-06-18_203252_0-0-0.dat",
        "Z0720-07m_BR_Sec22/Merlin-6281_21-06-18_203252_0-0-1.dat",
    ]
    dat_files_203350 = [
        "Z0720-07m_BR_Sec22/Merlin-6281_21-06-18_203350_0-0-0.dat",
        "Z0720-07m_BR_Sec22/Merlin-6281_21-06-18_203350_0-0-1.dat",
    ]
    dataset_groups = [
        dat_files_125416,
        dat_files_125527,
        dat_files_203058,
        dat_files_203155,
        dat_files_203252,
        dat_files_203350,
    ]
    return dataset_groups


def get_parameters():
    chunks_list = [
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (3500, 7437),
    ]
    compressors = []

    compressors.append(hdf5plugin.Bitshuffle())
    compressors.append(hdf5plugin.LZ4())
    compressors.append({"shuffle": True, **hdf5plugin.LZ4()})
    compressors.append(hdf5plugin.Blosc())

    for aggression in range(1, 10):
        compressors.append(
            {"compression": hdf5plugin.ZSTD_ID, "compression_opts": (aggression,)}
        )

    for aggression in range(1, 10):
        compressors.append({"compression": "gzip", "compression_opts": aggression})

    params = [{"chunks": None}]
    for chunks in chunks_list:
        for compressor in compressors:
            params.append({"chunks": chunks, **compressor})
    return params


def benchmark_fibsem_h5(
    groups=get_dat_file_groups(),
    params=get_parameters(),
    results_dir=os.path.join("results", datetime.now().isoformat()),
):
    #  Make the directory to save data in
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    groups_file = os.path.join(results_dir, "groups.json")
    params_file = os.path.join(results_dir, "params.json")
    results_file = os.path.join(results_dir, "results.json")
    partial_file = os.path.join(results_dir, "partial.json")

    # Save the groups and params we will iterate over
    with open(groups_file, "w") as f:
        json.dump(groups, f)
    with open(params_file, "w") as f:
        json.dump(params, f)

    # Results over groups
    results = []
    for group in groups:
        # Results over parameters
        group_results = []

        # Get file size of the original dat files
        datsize = 0
        for datfile in group:
            print(datfile)
            datsize += os.path.getsize(datfile)
        print(f"total filesize: %d" % datsize)

        for param in params:
            print("    ", param)

            # Write the HDF5 file
            write_h5_file = lambda: create_aggregate_fibsem_h5_file_from_dat_filenames(
                group, overwrite=True, **param
            )
            h5_file = write_h5_file()
            write_time = timeit(write_h5_file, number=1)

            # Read the HDF5 file
            read_h5_file = lambda: load_fibsem_from_h5_file(h5_file)
            read_time = timeit(read_h5_file, number=1)

            # Collect HDF5 file statistics
            size = os.path.getsize(h5_file)
            ratio = size / datsize

            # Report the results
            result = {
                "write_time": write_time,
                "read_time": read_time,
                "size": size,
                "ratio": ratio,
            }
            print("        ", result)

            # Save the results
            group_results.append(result)
            # Dump json line by line to partial file
            with open(partial_file, "a") as f:
                json.dump(result, f)
                print(file=f)

        results.append(group_results)

    # Save the entire results structure
    with open(results_file, "w") as f:
        json.dump(results, f)
    return results


if __name__ == "__main__":
    benchmark_fibsem_h5()
