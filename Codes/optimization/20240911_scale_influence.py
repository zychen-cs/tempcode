import math
from os import read
import queue
from codetiming import Timer
import asyncio
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from itertools import count
import time
from matplotlib.animation import FuncAnimation
from numpy.core.numeric import True_
import matplotlib
import queue
import asyncio
import struct
import os
import sys
import time
import datetime
import atexit
import time
import numpy as np
from bleak import BleakClient
import matplotlib.pyplot as plt
from bleak import exc
import pandas as pd
import atexit
from multiprocessing import Pool
import multiprocessing

from src.solver import Solver_jac, Solver
from src.filter import Magnet_KF, Magnet_UKF
from src.preprocess import Calibrate_Data
from config import pSensor_smt, pSensor_joint_exp,pSensor_large_smt, pSensor_small_smt, pSensor_median_smt, pSensor_imu, pSensor_ear_smt,pSensor_selfcare
import cppsolver as cs
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def identify_frequency(data, sampling_rate):
    # 计算FFT
    n = len(data)
    fft_values = fft(data)
    fft_values = np.abs(fft_values[:n//2])  # 取前半部分
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)[:n//2]  # 对应的频率

    # 绘制频谱
    plt.plot(freqs, fft_values)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

    # 找到最大频率分量
    peak_frequency = freqs[np.argmax(fft_values)]
    print(f"Identified dominant frequency: {peak_frequency} Hz")
    return peak_frequency
from scipy.signal import butter, filtfilt

def bandstop_filter(data, lowcut, highcut, sampling_rate, order=2):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandstop')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def lowpass_filter(data, cutoff, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    cutoff = cutoff / nyquist
    b, a = butter(order, cutoff, btype='low')
    filtered_data = filtfilt(b, a, data)
    return filtered_data



pSensor = pSensor_small_smt

params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
                   0, np.log(0.2), 1e-2 * (0), 1e-2 * (5), 1e-2 * (-0.17), np.pi, 0])

params2 = np.array([
    40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(0.08),
    1e-2 * 4, 1e-2 * 2, 1e-2 * (-0.17), 0, 0,
    1e-2 * 3, 1e-2 * 8, 1e-2 * (-0.17), 0, 0,
])
countnum=1
countnum1=1
resultslist=[]
resultslist1=[]


cali_path = '/home/czy/桌面/magx-main1/0911_I2C_8sensor_441_magnet_1.csv'
# cali_path1 = '/home/czy/桌面/magx-main1/0908_I2C_7sensor_406_1.csv'
name = ['Time Stamp', 'x',
        'y', 'z', 'theta', 'phy']
# name1 = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8','Sensor 9', 'Sensor 10','Sensor 11','Sensor 12' ]
name1 = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
        'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']
# name1=['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8',
#         'Sensor 9', 'Sensor 10','Sensor 11','Sensor 12',
#         'Sensor 13', 'Sensor 14','Sensor 15','Sensor 16']

'''The calculation and visualization process'''
t = 0
matplotlib.use('Qt5Agg')
# Nordic NUS characteristic for RX, which should be writable
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# Nordic NUS characteristic for TX, which should be readable
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
result = []
worklist = multiprocessing.Manager().Queue()

results = multiprocessing.Manager().Queue()
results2 = multiprocessing.Manager().Queue()
result_draw=[]

def end():
    print('End of the program')
    sys.exit(0)
def ang_convert(x):
    a = x//(2*np.pi)
    result = x-a*(2*np.pi)
    # if result > np.pi:
    #     result -= np.pi * 2
    if result <0:
        result += np.pi * 2
    return result

def ang_convert1(x):
    a = x//(2*np.pi)
    result = x-a*(2*np.pi)
    if result > np.pi:
        result -= np.pi
    if result <0:
        result += np.pi
    return result

def calculation_parallel():
    global worklist
    global params
    global params2
    global results
    global results2
    global pSensor
    global resultslist
    global resultslist1
    global countnum
    global countnum1
    global pSensor
    global worklist
    global resultslist1
    global countnum1
    # global direction
    myparams1 = params
    myparams2 = params2
       
    data = pd.read_csv("/home/czy/桌面/magx-main1/0911_I2C_8sensor_441_magnet_no.csv")
    
    num = 8


    all_data = []
    sensors = np.zeros((num, 3))
    current = [datetime.datetime.now()]
    calibration = np.load('result/calibration.npz')
    offset = calibration['offset'].reshape(-1)
    scale = calibration['scale'].reshape(-1)
    print(offset)
    print(scale)
    
   
    
    for j in range (0,len(data)):
        sensors = np.zeros((num, 3))
        for i in range(0,num):
            column_name = f"Sensor {i+1}"  # 构造列名
            datatemp = data[column_name][j]
            # print()
            datatemp = datatemp.strip("()")
            # 根据逗号分割字符串
            numbers = datatemp.split(", ")
            # 转换为浮点数
           
            sensors[i, 0] =  numbers[0]
            sensors[i, 1] =  numbers[1]
            sensors[i, 2] =  numbers[2]
            # print(sensors[i,0])
       
        sensors = sensors.reshape(-1)
        # print(sensors)
        # sensors = (sensors - offset) / scale * np.mean(scale)
        sensors = (sensors - offset)/scale *33
        # result_draw.append(sensors[20])
        result_draw.append(math.sqrt(sensors[18]**2+sensors[19]**2+sensors[20]**2))
        # result_draw.append(math.sqrt(sensors[0]**2+sensors[1]**2+sensors[2]**2))
    
    print(np.mean(result_draw))
    # 将12个数组绘制为折线图
    plt.figure(figsize=(10, 6))
    
    plt.plot(result_draw)

    # 设置x轴的刻度间隔为1
    # plt.xticks(range(len(offsets['offset1'])))  # 根据数组长度设置刻度

    # 添加图例、标题和标签
    # plt.legend()
    # plt.title('Values of Offsets')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

    plt.tight_layout()


        
                        
                
  




if __name__ == '__main__':

    if True:
        calibration = Calibrate_Data(cali_path)
        [offset, scale] = calibration.cali_result()
        if not os.path.exists('result'):
            os.makedirs('result')
        np.savez('result/calibration.npz', offset=offset, scale=scale)
        print(np.mean(scale))

    calculation_parallel() # For tracking 1 magnet
    # calculation_parallel(False) # For tracking 1 magnet
    # asyncio.run(main(2)) # For tracking 2 magnet

