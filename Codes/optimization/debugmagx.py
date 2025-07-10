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


cali_path = '/home/czy/桌面/magx-main1/0508_magx_1.csv'
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

def calculation_parallel(magcount=1, use_kf=0, use_wrist=False):
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
       
    data = pd.read_csv("/home/czy/桌面/magx-main1/debug1.csv")

    num = int(pSensor.size/3)


    all_data = []
    sensors = np.zeros((num, 3))
    current = [datetime.datetime.now()]
    calibration = np.load('result/calibration.npz')
    offset = calibration['offset'].reshape(-1)
    scale = calibration['scale'].reshape(-1)
    # print("offset",offset)
    # print("scale",scale)
   
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
        sensors = (sensors - offset) / scale * np.mean(scale)
        # print(sensors)
        if len(all_data) > 3:
            sensors = (sensors + all_data[-1] + all_data[-2]) / 3
        all_data.append(sensors)
        worklist.put(sensors)
        if not worklist.empty():
                datai = worklist.get()
                datai = datai.reshape(-1, 3)
                # resulti [gx, gy, gz, m, x0,y0,z0, theta0, phy0, x1, y1, z1, theta1, phy1]
                if magcount == 1:
                    if np.max(np.abs(myparams1[4:7])) > 1:
                        myparams1 = params
                    resulti = cs.solve_1mag(
                        datai.reshape(-1), pSensor.reshape(-1), myparams1)
                    myparams1 = resulti
                    # print(resulti[0])
                    # print(resulti[1])
                    # print(resulti[2])
                    result = [resulti[4] * 1e2,
                            resulti[5] * 1e2, resulti[6] * 1e2]
                    results.put(result)
                    current = [datetime.datetime.now()]
                    # direction=np.array([np.sin(ang_convert(resulti[7]))*np.cos(ang_convert(resulti[8])),
                    #           np.sin(ang_convert(resulti[7]))*np.sin(ang_convert(resulti[8])), np.cos(ang_convert(resulti[7]))])
                    current.append(resulti[4] * 1e2)
                    current.append(resulti[5] * 1e2)
                    current.append(resulti[6] * 1e2)
                    current.append(ang_convert1(resulti[7]))
                    current.append(ang_convert(resulti[8]))
                    # current.append((resulti[7]))
                    # current.append((resulti[8]))
                    resultslist.append(current)
                    # print(resultslist)
                    # resultlist.append(current)
                    # print(resulti[3])
                    print("Orientation: {:.2f}, {:.2f}, m={:.2f}".format(
                        resulti[7] / np.pi * 180,
                        resulti[8] / np.pi * 180 % 360,
                        np.exp(resulti[3])))
                    print("Position: {:.2f}, {:.2f}, {:.2f}, m={:.2f}, dis={:.2f},orientation: {:.2f},{:.2f}".format(
                        result[0],
                        result[1],
                        result[2],
                        resulti[3],
                        np.sqrt(
                            result[0] ** 2 + result[1] ** 2 + result[2] ** 2),
                        ang_convert1(resulti[7]),
                        ang_convert(resulti[8]))),
                        
                
  




if __name__ == '__main__':

    # if True:
    #     calibration = Calibrate_Data(cali_path)
    #     [offset, scale] = calibration.cali_result()
    #     if not os.path.exists('result'):
    #         os.makedirs('result')
    #     np.savez('result/calibration.npz', offset=offset, scale=scale)
    #     print(np.mean(scale))

    calculation_parallel() # For tracking 1 magnet
    # asyncio.run(main(2)) # For tracking 2 magnet

