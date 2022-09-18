import os
import numpy as np

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#     print(memory_available)
#     return 'cuda:'+str(np.argmax(memory_available))
    return 'cuda:1'
