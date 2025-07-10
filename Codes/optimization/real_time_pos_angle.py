from os import read
import queue
from codetiming import Timer
import asyncio
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
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
from config import pSensor_smt, pSensor_joint_exp,psensor_magdot,pSensor_large_smt, pSensor_small_smt, pSensor_median_smt, pSensor_imu, pSensor_ear_smt,pSensor_selfcare
import cppsolver as cs

'''The parameter user should change accordingly'''
# Change pSensor if a different sensor layout is used
# pSensor = pSensor_ear_smt
# pSensor = pSensor_selfcare
# pSensor = pSensor_ear_smt
# pSensor = psensor_magdot
pSensor = pSensor_joint_exp
# pSensor = pSensor_small_smt
# Change this parameter for differet initial value for 1 magnet

# 0.08 10*2
#0.003 5*0.3
#0.015 5*1
# 0.1  10*5*3
#0.17  10*5*5
#  0.2 球体
# 0.025 球体
#0.05 8*5*2
#  0.39 0.32球体 d=1cm
#  1.6 1.38球体 d=1.6cm
#1cm*1cm*1cm 0.77
# params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
#                    0, np.log(0.25), 1e-2 * (5), 1e-2 * (5), 1e-2 * (2), np.pi, np.pi])
#                 #    np.pi/2, np.pi
# params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
#                    0, np.log(0.25), 1e-2 * (4), 1e-2 * (2), 1e-2 * (-0.17), np.pi, 0])
params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
                   0, np.log(0.46), 1e-2 * (0), 1e-2 * (-5), 1e-2 * (-0.17), 0, 0])
#Gx: 0, 40,80,120,160,1000
#Gy:0, 40,80,120,160,1000
#Gy:0, 40,80,120,160,1000
#m:0.06,0.26,0.46,0.66,0.86
#x:5,10,-5,-10,0
#y:5,10,-5,-10,0
#z:5,10,-5,-10,0
#theta:pi,-pi,pi/2,-pi/2,0
#phy:
# params = np.array([40 * 1e-6, 0* 1e-6,
#                    0 * 1e-6, np.log(0.46), 1e-2 * (0), 1e-2 * (-5), 1e-2 * (0), 0, 0])
# params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
#                    0, np.log(0.2), 1e-2 * (1), 1e-2 * (2), 1e-2 * (-0.17), np.pi, 0])
                #    np.pi/2, np.pi
# Change this parameter for different initial value for 2 magnets
params2 = np.array([
    40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, np.log(0.08),
    1e-2 * 5, 1e-2 * 3, 1e-2 * (1), np.pi, 0,
    1e-2 * 2, 1e-2 * 8, 1e-2 * (-0.17), np.pi, 0,
])
countnum=1
countnum1=1
resultslist=[]
resultslist1=[]
# Your adafruit nrd52832 ble address
# address = ("FA:0A:21:CD:68:61")
# ble_address = "F0:33:85:3D:67:9D"
ble_address = "CD:7A:5F:1E:8B:07"
# ble_address = "CE:E1:85:67:A6:2B"
# ble_address = "EE:66:70:D4:74:5D"
# ble_address = "CC:42:8E:0E:D5:D5"
# Absolute or relative path to the calibration data, stored in CSV4
cali_path = '/home/czy/桌面/magx-main1/1030_I2C_8sensor_441_1.csv'
name = ['Time Stamp', 'Gx','Gy','Gz','x',
        'y', 'z', 'theta', 'phy']
# name1 = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8','Sensor 9', 'Sensor 10']
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

def calculate_distance(point1, point2):
    """计算两个点之间的欧几里得距离"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + 
                     (point1[1] - point2[1]) ** 2 + 
                     (point1[2] - point2[2]) ** 2)

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
    # global direction
    myparams1 = params
    myparams2 = params2
    while True:
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
                result1 = [resulti[4] * 1e2,
                          resulti[5] * 1e2]
                # 计算平方和
                square_sum = sum([x**2 for x in result1])

                # 取平方根
                dis1 = math.sqrt(square_sum)
                # dis1 = math.sqrt(resulti[4] * 1e2+resulti[5] * 1e2**2+resulti[6] * 1e2)
                dis2 = 3.2
                dis3 = 3
                # 计算并输出每个角度
                def calculate_angle(a, b, c):
                    # 使用余弦定理计算角度的余弦值
                    cos_theta = (b**2 + c**2 - a**2) / (2 * b * c)
                    
                    # 处理可能的数值超出范围的情况
                    if cos_theta < -1:
                        cos_theta = -1
                    elif cos_theta > 1:
                        cos_theta = 1
                    
                    try:
                        # 计算弧度值
                        theta = math.acos(cos_theta)
                        # 将弧度转为度
                        theta_deg = math.degrees(theta)
                        return theta_deg
                    except ValueError:
                        # 如果计算失败，默认返回0度
                        return 0

                # 计算第一个角度
                theta1_deg = calculate_angle(dis1, dis2, dis3)
                print(f"Angle 1 (theta1): {theta1_deg:.2f} degrees")

                # 计算第二个角度
                theta2_deg = calculate_angle(dis2, dis1, dis3)
                print(f"Angle 2 (theta2): {theta2_deg:.2f} degrees")

                # 计算第三个角度
                theta3_deg = calculate_angle(dis3, dis1, dis2)
                print(f"Angle 3 (theta3): {theta3_deg:.2f} degrees")
                results.put(result)
                current = [datetime.datetime.now()]
                # direction=np.array([np.sin(ang_convert(resulti[7]))*np.cos(ang_convert(resulti[8])),
                #           np.sin(ang_convert(resulti[7]))*np.sin(ang_convert(resulti[8])), np.cos(ang_convert(resulti[7]))])
                current.append(resulti[0]*1e6)
                current.append(resulti[1]*1e6)
                current.append(resulti[2]*1e6)
                current.append(resulti[4] * 1e2)
                current.append(resulti[5] * 1e2)
                current.append(resulti[6] * 1e2)
                current.append(ang_convert1(resulti[7]))
                current.append(ang_convert(resulti[8]))
                # current.append((resulti[7]))
                # current.append((resulti[8]))
                resultslist.append(current)
                # print(resultslist)
                # if(len(resultslist)==1000):
                #      print("Output csv")
                #      test = pd.DataFrame(columns=name,data=resultslist)
                #      test.to_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/1030leap1_"+str(countnum)+".csv")
                #      print("Exited")
                #      countnum=countnum+1
                #      resultslist=[]
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
                    
                
            elif magcount == 2:
                if np.max(
                        np.abs(myparams2[4: 7])) > 1 or np.max(
                        np.abs(myparams2[9: 12])) > 1:
                    myparams2 = params2

                resulti = cs.solve_2mag(
                    datai.reshape(-1), pSensor.reshape(-1), myparams2)
                
                result = [resulti[4] * 1e2,
                          resulti[5] * 1e2, resulti[6] * 1e2]
                sensor_pos=[1.2192, 5.08, -7]
                result2 = [resulti[9] * 1e2,
                           resulti[10] * 1e2, resulti[11] * 1e2]
                dis1 = calculate_distance(result,sensor_pos)
                dis2 = calculate_distance(result2,sensor_pos)
                if(dis1<dis2):
                    
                    results.put(result)
                    results2.put(result2)
                else:
                    temp=result2
                    
                     
                    results.put(result2)
                    results2.put(result)
                    result2 = result
                    result = temp

                    temp1=resulti[9]
                    temp2=resulti[10]
                    temp3=resulti[11]
                    resulti[9]=resulti[4]
                    resulti[10]=resulti[5]
                    resulti[11]=resulti[6]
                    resulti[4]=temp1
                    resulti[5]=temp2
                    resulti[6]=temp3

                myparams2 = resulti

                
                
                
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
                if(len(resultslist)==3000):
                     print("Output csv")
                     test = pd.DataFrame(columns=name,data=resultslist)
                     test.to_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0911test__"+str(countnum)+".csv")
                     print("Exited")
                     countnum=countnum+1
                     resultslist=[]
                
                current1 = [datetime.datetime.now()]
                # direction=np.array([np.sin(ang_convert(resulti[7]))*np.cos(ang_convert(resulti[8])),
                #           np.sin(ang_convert(resulti[7]))*np.sin(ang_convert(resulti[8])), np.cos(ang_convert(resulti[7]))])
                current1.append(resulti[9] * 1e2)
                current1.append(resulti[10] * 1e2)
                current1.append(resulti[11] * 1e2)
                current1.append(ang_convert1(resulti[12]))
                current1.append(ang_convert(resulti[13]))
                # current.append((resulti[7]))
                # current.append((resulti[8]))
                resultslist1.append(current1)
                # print(resultslist)
                if(len(resultslist1)==3000):
                     print("Output csv")
                     test = pd.DataFrame(columns=name,data=resultslist1)
                     test.to_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/sim0911test_"+str(countnum1)+".csv")
                     print("Exited")
                     countnum1=countnum1+1
                     resultslist1=[]
                print(
                    "Mag 1 Position: {:.2f}, {:.2f}, {:.2f}, orientation: {:.2f},{:.2f},dis={:.2f} \n Mag 2 Position: {:.2f}, {:.2f}, {:.2f}, orientation: {:.2f},{:.2f},dis={:.2f}". format(
                        result[0],
                        result[1],
                        result[2],
                        ang_convert1(resulti[7]),
                        ang_convert(resulti[8]),             
                        np.sqrt(
                            result[0] ** 2 +
                            result[1] ** 2 +
                            result[2] ** 2),
                        result2[0],
                        result2[1],
                        result2[2],
                        ang_convert1(resulti[12]),
                        ang_convert(resulti[13]),  
                        np.sqrt(
                            result2[0] ** 2 +
                            result2[1] ** 2 +
                            result2[2] ** 2)))



async def task(name, work_queue):
    timer = Timer(text=f"Task {name} elapsed time: {{: .1f}}")
    while not work_queue.empty():
        delay = await work_queue.get()
        print(f"Task {name} running")
        timer.start()
        await asyncio.sleep(delay)
        timer.stop()

async def show_mag(magcount=1):
    global t
    global pSensor
    global results
    global results2
    # global direction
    myresults = np.array([[0, 0, 10]])
    myresults2 = np.array([[0, 0, 10]])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')

    # TODO: add title
    ax.set_xlabel('x(cm)')
    ax.set_ylabel('y(cm)')
    ax.set_zlabel('z(cm)')
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-10, 40])
    Xs = 1e2 * pSensor[:, 0]
    Ys = 1e2 * pSensor[:, 1]
    Zs = 1e2 * pSensor[:, 2]

    XXs = Xs
    YYs = Ys
    ZZs = Zs
    ax.scatter(XXs, YYs, ZZs, c='r', s=1, alpha=0.5)

    (magnet_pos,) = ax.plot(t / 100.0 * 5, t / 100.0 * 5, t /
                            100.0 * 5, linewidth=3, animated=True)
    if magcount == 2:
        (magnet_pos2,) = ax.plot(t / 100.0 * 5, t / 100.0 * 5, t /
                                 100.0 * 5, linewidth=3, animated=True)
    plt.show(block=False)
    plt.pause(0.1)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(magnet_pos)
    fig.canvas.blit(fig.bbox)
    # timer = Timer(text=f"frame elapsed time: {{: .5f}}")

    while True:
        # timer.start()
        fig.canvas.restore_region(bg)
        # update the artist, neither the canvas state nor the screen have
        # changed

        # update myresults
        if not results.empty():
            myresult = results.get()
            myresults = np.concatenate(
                [myresults, np.array(myresult).reshape(1, -1)])

        if myresults.shape[0] > 30:
            myresults = myresults[-30:]

        x = myresults[:, 0]
        y = myresults[:, 1]
        z = myresults[:, 2]

        xx = x
        yy = y
        zz = z

        magnet_pos.set_xdata(xx)
        magnet_pos.set_ydata(yy)
        magnet_pos.set_3d_properties(zz, zdir='z')
        # re-render the artist, updating the canvas state, but not the screen
        ax.draw_artist(magnet_pos)

        if magcount == 2:
            if not results2.empty():
                myresult2 = results2.get()
                myresults2 = np.concatenate(
                    [myresults2, np.array(myresult2).reshape(1, -1)])

            if myresults2.shape[0] > 30:
                myresults2 = myresults2[-30:]
            x = myresults2[:, 0]
            y = myresults2[:, 1]
            z = myresults2[:, 2]

            xx = x
            yy = y
            zz = z

            magnet_pos2.set_xdata(xx)
            magnet_pos2.set_ydata(yy)
            magnet_pos2.set_3d_properties(zz, zdir='z')
            ax.draw_artist(magnet_pos2)

        # copy the image to the GUI state, but screen might not changed yet
        fig.canvas.blit(fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        fig.canvas.flush_events()
        await asyncio.sleep(0)
        # timer.stop()


def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    global pSensor
    global worklist
    global resultslist1
    global countnum1
    num = int(pSensor.size/3)


    all_data = []
    sensors = np.zeros((num, 3))
    current = [datetime.datetime.now()]
    calibration = np.load('result/calibration.npz')
    offset = calibration['offset'].reshape(-1)
    scale = calibration['scale'].reshape(-1)
    # print("offset",offset)
    # print("scale",scale)
    for i in range(0,num):
        sensors[i, 0] =  struct.unpack('f', data[12 * i: 12 * i + 4])[0]
        sensors[i, 1] =  struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
        sensors[i, 2] =  struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
        # print("Sensor " + str(i+1)+": " +
        #       str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2]))
        # current.append(
        #     "("+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2])+")")
    # for i in range(0,num/2):
    #     sensors[i, 1] = -struct.unpack('f', data[12 * i: 12 * i + 4])[0]
    #     sensors[i, 2] = -struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
    #     sensors[i, 0] =  struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
    #     # print("Sensor " + str(i+1)+": " +
    #         #   str(sensors[i, 2]) + ", " + str(sensors[i, 0]) + ", " + str(sensors[i, 1]))
    #     current.append(
    #         "("+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2])+")")
    # for i in range(num/2,num):
    #     sensors[i, 1] =  struct.unpack('f', data[12 * i: 12 * i + 4])[0]
    #     sensors[i, 2] = -struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
    #     sensors[i, 0] = -struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
    #     # print("Sensor " + str(i+1)+": " +
    #     #       str(sensors[i, 2]) + ", " + str(sensors[i, 0]) + ", " + str(sensors[i, 1]))
        current.append(
            "("+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2])+")")
    #     # battery_voltage = struct.unpack('f', data[12 * num: 12 * num + 4])[0]
    #     # print("Battery voltage: " + str(battery_voltage))
    
    resultslist1.append(current)
    
    # if(len(resultslist1)==1000):
    #     print("Output csv")
    #     test = pd.DataFrame(columns=name1,data=resultslist1)
    #     test.to_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/1030leapsensor1_"+str(countnum1)+".csv")
    #     print("Exited")
    #     countnum1=countnum1+1
    #     resultslist1=[]
    sensors = sensors.reshape(-1)
    # print("=====================")
    # print(sensors)
    sensors = (sensors - offset)
    # sensors = (sensors - offset) / scale * 33
    # sensors = sensors - offset
    # print(sensors)
    # print("=====================")
    # print(sensors)
# for the ear ver of MagX; we need to unify the coordinates different sensors.(Since the sensors on the left ear are reversed in the X&Z)




    # print(len(all_data))
    if len(all_data) > 3:
        sensors = (sensors + all_data[-1] + all_data[-2]) / 3
        # print(sensors)
    all_data.append(sensors)
    # print(len(all_data))
    worklist.put(sensors)
    # print("############")



async def run_ble(address, loop):
    async with BleakClient(address, loop=loop) as client:
        # wait for BLE client to be connected
        x = await client.is_connected()
        print("Connected: {0}".format(x))
        print("Press Enter to quit...")
        # wait for data to be sent from client
        await client.start_notify(UART_TX_UUID, notification_handler)
        while True:
            await asyncio.sleep(0.01)
            # data = await client.read_gatt_char(UART_TX_UUID)


async def main(magcount=1):
    """
    This is the main entry point for the program
    """
    # Address of the BLE device
    global ble_address
    address = (ble_address)

    # Run the tasks
    with Timer(text="\nTotal elapsed time: {:.1f}"):
        multiprocessing.Process(
            target=calculation_parallel, args=(magcount, 1, False)).start()
        await asyncio.gather(
            asyncio.create_task(run_ble(address, asyncio.get_event_loop())),
            asyncio.create_task(show_mag(magcount)),
        )


if __name__ == '__main__':

    if True:
        calibration = Calibrate_Data(cali_path)
        [offset, scale] = calibration.cali_result()
        if not os.path.exists('result'):
            os.makedirs('result')
        np.savez('result/calibration.npz', offset=offset, scale=scale)
        print(np.mean(scale))

    asyncio.run(main(1))  # For tracking 1 magnet
    # asyncio.run(main(2)) # For tracking 2 magnet

