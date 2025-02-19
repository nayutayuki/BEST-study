'''
@Date: 2019-12-05 04:24:15
@Author: Yong Pi
@LastEditors: Yong Pi
@LastEditTime: 2019-12-05 04:24:15
@Description: All rights reserved.
'''
import os
import pynvml

def get_free_id():
    pynvml.nvmlInit()

    def get_free_ratio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5 * (float(use.gpu + float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if get_free_ratio(i) < 50:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus + str(g) + ','
    gpus = gpus[:-1]
    return gpus


def set_gpu_with_rate(gpu_input):
    freeids = get_free_id()
    if gpu_input == 'all':
        gpus = freeids
    else:
        gpus = gpu_input
        if any([g not in freeids for g in gpus.split(',')]):
            raise ValueError('gpu ' + g + ' is being used')
    print('using gpu ' + gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))


def get_device_id():
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus + str(g) + ','
    gpus = gpus[:-1]
    return gpus


def set_gpu(gpu_input):
    # import pdb;pdb.set_trace()
    if gpu_input == 'all':
        gpus = get_device_id()
    else:
        gpus = gpu_input
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))
