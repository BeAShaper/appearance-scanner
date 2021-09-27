import numpy as np
import sys
if __name__ == "__main__":
    thread_num = int(sys.argv[1])
    data_root = sys.argv[2]

    with open(data_root+"fitted.bin","wb") as f:
        for wich_thread in range(thread_num):
            tmp = np.fromfile(data_root+"fitted_{}.bin".format(wich_thread),np.float32)
            tmp.tofile(f)
    