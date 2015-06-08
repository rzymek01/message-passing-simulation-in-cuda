#!/usr/bin/python
import sys
import glob

OUTPUT_PATH = "/macierz/home/137396rm/cuda/msg-pass-sim-tests/"


def generate_csv(path, args):
    time_list, thread_numbers = get_time_from_files(path, args)
    # print time_list
    # place for device and
    # print thread_numbers

    for row in time_list:
        print(','.join([str(x) for x in row]))

def get_time_from_files(path, args):
    params = {
        "test_id": args['test_id'] if 'test_id' in args else '*',
        "v": args['v'] if 'v' in args else '*',
        "d": args['d'] if 'd' in args else '*',
        "t": args['t'] if 't' in args else '*',
        "test": args['test'] if 'test' in args else '*',
    }
    files = glob.glob(path + "test_" + params['test_id'] + "_" + params['v'] + "_" + params['d'] + "_" + params['t'] + "_" + params['test'] + ".time")
    # print files

    thread_numbers = set()
    time_list = list()

    column_id = 1
    columns = 0

    # i = 0
    for filename in files:
        test_param_list, test_ext = filename.split('.')
        test_param_list = test_param_list.split('_')
        file_params = {
            "test_id": test_param_list[1],
            "v": test_param_list[2],
            "d": test_param_list[3],
            "t": test_param_list[4],
            "test": test_param_list[5],
        }
        test_param_list = []
        for key in file_params:
            if '*' == params[key]:
              test_param_list.append(file_params[key])

        columns = len(test_param_list)

        # append to list time from file
        with open(filename, 'r') as f:
            time = f.readline()

        if len(time) == 0:
            time = 0
        test_param_list.append(float(time))
        # print test_param_list
        # add thread number to set
        thread_numbers.add(test_param_list[column_id])

        time_list.append(test_param_list)

    def my_sort(x):
        t = ()
        for i in range(0, columns):
            if i != column_id:
              t = t + (x[i],)
        t = t + (x[column_id],)
        return t

    time_list.sort(key=my_sort)
    # print time_list

    return time_list, len(thread_numbers)


if __name__ == "__main__":
    # argc = len(sys.argv)
    # if argc == 3:
    #     test_id = sys.argv[1]
    #     v = str(sys.argv[2])
    # else:
    #     print("usage: " + sys.argv[0] + " <test_id> <version>")
    #     exit()

    params = {"test_id": "B", "t": "1024", "d": "0"}

    path = OUTPUT_PATH
    generate_csv(path, params)
