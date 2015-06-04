import sys
import glob

OUTPUT_PATH = "/macierz/home/137396rm/cuda/msg-pass-sim/output/"


def generate_csv(path, test_id, v):
    time_list, thread_numbers = get_time_from_files(path, test_id, v)
    # print time_list
    # place for device and
    # print thread_numbers

    content = ""

    time_list_len = len(time_list)
    i = 0
    while i < time_list_len:
        content += str(time_list[i][0]) + "," + str(time_list[i][2]) + ","
        for j in range(0, thread_numbers):
            content += str(time_list[i][3])
            if j != thread_numbers - 1:
                content += ","
            i += 1

        content += "\n"

    print content


def get_time_from_files(path, test_id, v):
    files = glob.glob(path + "test_" + test_id + "_" + v + "*.time")
    # print files

    thread_numbers = set()
    time_list = list()
    time_dict = dict()

    # i = 0
    for filename in files:
        test_param_list, test_ext = filename.split('.')
        test_param_list = test_param_list.split('_')
        del test_param_list[0:3]
        # test_param_list = ['2', '1024', '1000']
        test_param_list[0] = int(test_param_list[0])
        test_param_list[1] = int(test_param_list[1])
        test_param_list[2] = int(test_param_list[2])

        # append to list time from file
        with open(filename, 'r') as f:
            time = f.readline()

        if len(time) == 0:
            time = 0
        test_param_list.append(float(time))
        # print test_param_list
        # add thread number to set
        thread_numbers.add(test_param_list[1])

        time_list.append(test_param_list)
        # convert lists to dict
        # time_dict[i] = {test_param_list[2]: {test_param_list[4]: {test_param_list[3]: test_param_list[5]}}}
        # i += 1
    time_list.sort(key=lambda x: (x[0], x[2], x[1]))
    # print time_list

    return time_list, len(thread_numbers)


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc == 3:
        test_id = sys.argv[1]
        v = str(sys.argv[2])
    else:
        print("usage: " + sys.argv[0] + " <test_id> <version>")
        exit()

    path = OUTPUT_PATH
    generate_csv(path, test_id, v)
