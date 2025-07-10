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
from sklearn.preprocessing import StandardScaler
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
import joblib

class Calibrate_Data1:
    def __init__(self, calibration_data):
        super().__init__()
        self.calibration_data = calibration_data
        self.nSensor = 8  # Assuming you have 8 sensors
        self.build_dict()

    def build_dict(self):
        tstamps = []
        readings = []

        for entry in self.calibration_data:
            tstamp = entry[0]  # The timestamp
            # Collect sensor readings by parsing each sensor reading
            sensor_readings = [self.parse_sensor_reading(reading) for reading in entry[1:]]
            tstamps.append(tstamp)
            readings.append(sensor_readings)

        self.tstamps = np.array(tstamps)
        self.raw_readings = np.array(readings)

        # Sort the data according to the time stamp
        index = np.argsort(self.tstamps)
        self.tstamps = self.tstamps[index]
        self.raw_readings = self.raw_readings[index]

        # Reshape to a 10 x 24 array or whatever shape is required
        self.readings = self.reshape_readings(self.raw_readings)

    def parse_sensor_reading(self, reading):
        # Directly return the numpy array from the tuple
        return np.array(reading)

    def reshape_readings(self, readings):
        # Flatten the sensor readings for each timestamp
        readings_flat = [r.flatten() for r in readings]
        # Convert to a numpy array
        readings_array = np.array(readings_flat)
        
        # Ensure the array has the right shape (adjust the shape as necessary)
        # For 10 timestamps with 24 sensor readings, this is just an example
        if readings_array.shape[0] > 10:
            return readings_array[:10, :].reshape(-1, 24)  # Example to take only 10 entries
        return readings_array.reshape(-1, 24)  # Reshape according to actual data dimensions

'''The parameter user should change accordingly'''
# Change pSensor if a different sensor layout is used
# pSensor = pSensor_ear_smt
# pSensor = pSensor_selfcare
# pSensor = pSensor_ear_smt
# pSensor = psensor_magdot
pSensor = pSensor_joint_exp

# set magnet parameters
params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
                   0, np.log(0.46), 1e-2 * (0), 1e-2 * (-5), 1e-2 * (-0.17), 0, 0])
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
cali_path = '/home/czy/桌面/magx-main1/1020_I2C_8sensor_427_1.csv'
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
scale=[]
calibrationflag= False
matplotlib.use('Qt5Agg')
# Nordic NUS characteristic for RX, which should be writable
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# Nordic NUS characteristic for TX, which should be readable
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
result = []
worklist = multiprocessing.Manager().Queue()
calibration_queue = multiprocessing.Manager().Queue()
calibration_maxlen = 10  # 定义窗口最大长度
calibration_coefficient = None  # 用于存储校准系数

# 全局变量，用于保存上次校准的时间
last_calibration_time = time.time()  # 初始化为当前时间
calibration_interval = 1 * 30  # 校准时间间隔，5分钟（300秒）

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

# 校准算法（需要10个数据）
def calibrate(data_window):
    # 校准算法的逻辑，基于窗口的数据
    calibration_coefficient = np.mean(data_window, axis=0)  # 示例校准逻辑
    print(f"Calibrating with window: {data_window}")
    return calibration_coefficient

# 定义一个函数来计算每列的特征
def calculate_features(column_data):
    mean = np.mean(column_data)
    peak_to_peak = np.ptp(column_data)
    rms = np.sqrt(np.mean(column_data**2))
    std = np.std(column_data)
    q75, q25 = np.percentile(column_data, [75, 25])
    iqr = q75 - q25  # Interquartile range (IQR)
    
    return {
        'mean': mean,
        'std': std,
        'peak_to_peak': peak_to_peak,
        'rms': rms,
        
        'iqr': iqr
    }

# 维护一个滚动窗口，移除一部分旧数据
def maintain_rolling_window(calibration_queue, overlap_fraction=0.5):
    remove_count = int((1 - overlap_fraction) * calibration_maxlen)  # 每次移除的数量
    for _ in range(remove_count):
        calibration_queue.get()  # 移除最旧的数据

def calculation_parallel(magcount=1, use_kf=0, use_wrist=False):
    global worklist
    global calibration_queue
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
                    if(len(resultslist)==1000):
                        print("Output csv")
                        test = pd.DataFrame(columns=name,data=resultslist)
                        test.to_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/1020test2_"+str(countnum)+".csv")
                        print("Exited")
                        countnum=countnum+1
                        resultslist=[]
                    
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

# 9. 定义函数来预测新数据的类别并输出反标准化后的中心点
def predict_new_data(new_data, scaler_path, kmeans_path):
    # 加载保存的scaler和KMeans模型
    scaler = joblib.load(scaler_path)
    kmeans_model = joblib.load(kmeans_path)
    
    # 对新数据进行标准化
    new_data_scaled = scaler.transform(new_data)
    
    # 计算新数据到每个聚类中心的欧式距离
    distances = np.linalg.norm(kmeans_model.cluster_centers_ - new_data_scaled, axis=1)
    
    # 获取距离最近的聚类中心对应的类别
    predicted_cluster = np.argmin(distances)

    # 获取对应聚类中心并进行反标准化
    cluster_center = kmeans_model.cluster_centers_[predicted_cluster].reshape(1, -1)
    cluster_center_original_scale = scaler.inverse_transform(cluster_center)

    return predicted_cluster, cluster_center_original_scale

def notification_handler(sender, data):
    global pSensor
    global worklist
    global resultslist1
    global countnum1
    global calibrationflag
    global scale
    global last_calibration_time
    global calibration_interval
    num = int(pSensor.size / 3)

    sensors = np.zeros((num, 3))
    current = [datetime.datetime.now()]
    all_data = []
    # Load calibration data
    calibration = np.load('result/calibration.npz')
    offset = calibration['offset'].reshape(-1)
    # scale = calibration['scale'].reshape(-1)
    
    for i in range(num):
        sensors[i, 0] = struct.unpack('f', data[12 * i: 12 * i + 4])[0]
        sensors[i, 1] = struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
        sensors[i, 2] = struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
        current.append((sensors[i, 0], sensors[i, 1], sensors[i, 2]))  # Append actual sensor readings

    # print(current)  # This will now show the timestamp and the actual readings as tuples
    calibration_queue.put(current)

    # Maintain rolling window effect
    if calibration_queue.qsize() > calibration_maxlen:
        calibration_queue.get()  # Remove the oldest data


    # 计算自上次校准以来经过的时间
    current_time = time.time()
    time_since_last_calibration = current_time - last_calibration_time

    # Perform calibration when the queue is full
    if (not calibrationflag or time_since_last_calibration >= calibration_interval) and calibration_queue.qsize() == calibration_maxlen:
    # if calibration_queue.qsize() == calibration_maxlen and calibrationflag== False:
        calibration_data = []
        # calibration = list(calibration_queue.queue)
        for i in range(calibration_maxlen):
            calibration_data.append(calibration_queue.get())
        
        # Convert list of tuples to a numpy array
        calibration_data = np.array(calibration_data)
        # print("========================")
        # print(calibration_data)  # Now this should be numeric data
        # print("========================")

        calidata = Calibrate_Data1(calibration_data)
        print("This is raw data")
        rawdata = calidata.readings
        # 加载模型和标准化器
       # 加载模型和标准化器

        
        multi_target_forest = joblib.load('/media/czy/T7 Shield/ubuntu/calibration_project/multi_target_forest_model.pkl')
        scaler_X = joblib.load('/media/czy/T7 Shield/ubuntu/calibration_project/scaler_X.pkl')
        scaler_Y = joblib.load('/media/czy/T7 Shield/ubuntu/calibration_project/scaler_Y.pkl')
        
        
        # 对输入数据进行标准化
        X_new_scaled = scaler_X.transform(rawdata)  # 注意这里使用的是transform，而不是fit_transform

        # 进行预测
        Y_new_pred_scaled = multi_target_forest.predict(X_new_scaled)

        # 对预测结果进行逆转换
        Y_new_pred_rescaled = scaler_Y.inverse_transform(Y_new_pred_scaled)

        print("********************")
        print(Y_new_pred_rescaled)
        print("********************")

        G_scalerX = joblib.load('/home/czy/桌面/calibartion_free/1016model/scaler_X.pkl')
        G_scalerXmodel = joblib.load('/home/czy/桌面/calibartion_free/1016model/scale2.pkl')
        # 对每一列计算特征
        # 计算每一列的均值
        
        mean_values = np.mean(Y_new_pred_rescaled, axis=0)
        mean_values= np.repeat(mean_values, 3)
        print("********************")
        print(mean_values)
        print("********************")
        # 将均值作为 new_data
        new_data = mean_values.reshape(1, -1)  # 转换为 (1, n) 形状，适应模型输入
        print

        # 使用模型进行预测
        predicted_label, scale = predict_new_data(new_data, '/media/czy/T7 Shield/ubuntu/calibration_project/kmeansscaler_model.pkl', '/media/czy/T7 Shield/ubuntu/calibration_project/kmeans_model.pkl')

        print(f'Predicted Cluster for new data: {predicted_label}')
        print(f'Original scale of the predicted cluster center: {scale}')
        

        # 打印预测结果
        # print("预测结果：", scale)
        # scale = np.repeat(scale, 3)
        calibrationflag = True
        last_calibration_time = current_time  # 更新上次校准时间
        # maintain_rolling_window(calibration_queue, overlap_fraction=0.5)
        # Manually put back the last 50% of the data into the queue
        for data in calibration_data[int(calibration_maxlen * 0.5):]:
            calibration_queue.put(data)
    if (calibrationflag==True):
        sensors = sensors.reshape(-1)
        # print("=====================")
        # print(sensors)
        # sensors = (sensors - offset) / scale * np.mean(scale)
        sensors = (sensors - offset) / scale * 33
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
        [offset, _] = calibration.cali_result()
        if not os.path.exists('result'):
            os.makedirs('result')
        np.savez('result/calibration.npz', offset=offset)
        # print(np.mean(scale))

    asyncio.run(main(1))  # For tracking 1 magnet
    # asyncio.run(main(2)) # For tracking 2 magnet

