#!/usr/bin/python
import subprocess
import os

print os.getcwd()

OUTPUT_PATH = "/macierz/home/137396rm/cuda/msg-pass-sim/output/"
INPUT_PATH = "/macierz/home/137396rm/cuda/msg-pass-sim/test/"
PROGRAM_PATH = "/macierz/home/137396rm/cuda/msg-pass-sim/Release/network"


def execute(specs_id, v, t, d, test):
    cmd = PROGRAM_PATH + str(v) + " " + str(t) + " " + str(d) + " <" + INPUT_PATH + "test" + str(test) + ".in >"\
        + OUTPUT_PATH + "test_" + specs_id + "_" + str(v) + "_" + str(d) + "_" + str(t) + "_" + str(test) + ".out 2>"\
        + OUTPUT_PATH + "test_" + specs_id + "_" + str(v) + "_" + str(d) + "_" + str(t) + "_" + str(test) + ".time"

    subprocess.call(cmd, shell=True)
#    print(cmd)


def run(specs):
    for i in specs:
        # print(i)
        specs_id = specs[i]["id"]
        specs_name = specs[i]["name"]
        specs_cmd = specs[i]["cmd"]

        for t in specs_cmd["t"]:
            # print(t)
            for d in specs_cmd["d"]:
                # print(d)
                for v in specs_cmd["v"]:
                    # print(v)
                    for test in specs_cmd["test"]:
                        # print(test)
                        execute(specs_id, v, t, d, test)


def find_diff(specs, v1, v2):
    for i in specs:
        # print(i)
        specs_id = specs[i]["id"]
        specs_name = specs[i]["name"]
        specs_cmd = specs[i]["cmd"]
        specs_v = specs[i]["cmd"]["v"]

        specs_v_len = len(specs_v)
        if specs_v_len == 1:
            # print "nothing to do in diff"
            return
        else:
            # print "check diff"
            for t in specs_cmd["t"]:
                # print(t)
                for d in specs_cmd["d"]:
                    # print(d)
                    for test in specs_cmd["test"]:
                        # print(test)
                        path = OUTPUT_PATH
                        diff_cmd = "diff -q " + path + "test_"\
                                   + specs_id + "_" + str(v1) + "_" + str(d) + "_" + str(t) + "_" + str(test)\
                                   + ".out "\
                                   + path + "test_"\
                                   + specs_id + "_" + str(v2) + "_" + str(d) + "_" + str(t) + "_" + str(test)\
                                   + ".out"
                        try:
                            diff_output = subprocess.check_output(diff_cmd, shell=True)
                        except subprocess.CalledProcessError as e:
                            print e.output
                            exit("Outputs are different. Something wrong with program.")

# main
if __name__ == "__main__":
    most_optimized_version = 1
    faster_device = 0

    specs = {
        # "A": {
        #     "id": "A",
        #     "name": "Test A. Execution time vs number of threads per block",
        #     "cmd": {
        #         "t": (256, 512, 1024),
        #         "d": (faster_device,),
        #         "v": (most_optimized_version,),
        #         "test": (1000,)
        #     }
        # },
        "B": {
            "id": "B",
            "name": "Test B. Execution time vs optimizations",
            "cmd": {
                "t": (1024,),
                "d": (faster_device,),
                "v": (1, most_optimized_version),
                "test": (10, 100, 1000, 10000)
            }
        },
        # "C": {
        #     "id": "C",
        #     "name": "Test C. Execution time vs size of input data (number of vertices)",
        #     "cmd": {
        #         "t": (1024,),
        #         "d": (faster_device,),
        #         "v": (most_optimized_version,),
        #         "test": (10, 100, 1000, 10000, '10b', '100b', '1000b', '10000b')
        #     }
        # },
        # "D": {
        #     "id": "D",
        #     "name": "Test D. Execution time vs device",
        #     "cmd": {
        #         "t": (1024,),
        #         "d": (0, 1),
        #         "v": (most_optimized_version,),
        #         "test": (1000, '1000b')
        #     }
        # },
        # "E": {
        #     "id": "E",
        #     "name": "Test E. Execution time vs size of input data, number of threads per block and optimizations",
        #     "cmd": {
        #         "t": (256, 512, 1024),
        #         "d": (faster_device,),
        #         "v": (1, most_optimized_version),
        #         "test": ('10b', '100b', '1000b', '10000b')
        #     }
        # },
        # "F": {
        #     "id": "F",
        #     "name": "Test F. Execution time vs size of input data, number of threads per block and device",
        #     "cmd": {
        #         "t": (256, 512, 1024),
        #         "d": (0, 1),
        #         "v": (most_optimized_version,),
        #         "test": ('10b', '100b', '1000b', '10000b')
        #     }
        # },
    }

    run(specs)
    find_diff(specs, 1, most_optimized_version)
