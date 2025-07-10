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
from src.preprocess import Calibrate_Data,Reading_Data
from config import pSensor_smt, pSensor_joint_exp,psensor_magdot,pSensor_large_smt, pSensor_small_smt, pSensor_median_smt, pSensor_imu, pSensor_ear_smt,pSensor_selfcare
import cppsolver as cs
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


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
cali_path = '/home/czy/桌面/magx-main1/0423cali.csv'
name = ['Time Stamp', 'x',
        'y', 'z', 'runtime']
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
from collections import deque

# 队列存储最新的100条数据
data_queue = deque(maxlen=100)
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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # 输入层到第一个隐藏层
        self.fc1 = nn.Linear(84, 256)  # 输入 21 维，扩大到 256
        self.bn1 = nn.BatchNorm1d(256)  # 批归一化
        
        # 第二个隐藏层
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        # 第三个隐藏层
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # 第四个隐藏层
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        # 第五个隐藏层
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        
        # Dropout层
        self.dropout = nn.Dropout(0.3)
        
        # 输出层
        self.fc6 = nn.Linear(32, 5)  # 假设有 5 个类别
        
        # 添加残差连接
        self.residual_fc1 = nn.Linear(256, 128)
        self.residual_fc2 = nn.Linear(128, 64)

    def forward(self, x):
        # 第一层：全连接 + 激活 + 批归一化
        x1 = F.gelu(self.bn1(self.fc1(x)))
        
        # 第二层：全连接 + 残差连接 + 激活 + 批归一化
        residual1 = self.residual_fc1(x1)
        x2 = F.gelu(self.bn2(self.fc2(x1)) + residual1)
        
        # 第三层：全连接 + 激活 + 批归一化
        x3 = F.gelu(self.bn3(self.fc3(x2)))
        
        # 第四层：全连接 + 残差连接 + 激活 + 批归一化
        residual2 = self.residual_fc2(x2)
        x4 = F.gelu(self.bn4(self.fc4(x3)) + residual2)
        
        # 第五层：全连接 + 激活 + 批归一化
        x5 = self.dropout(F.gelu(self.bn5(self.fc5(x4))))
        
        # 输出层
        x_out = self.fc6(x5)
        return x_out

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        # print("Input:", x.shape)  # [B, 1, 84]

        x1 = self.inc(x)
        # print("x1:", x1.shape)

        x2 = self.down1(x1)
        # print("x2:", x2.shape)

        x3 = self.down2(x2)
        # print("x3:", x3.shape)

        x4 = self.down3(x3)
        # print("x4:", x4.shape)

        x5 = self.down4(x4)
        # print("x5:", x5.shape)

        x = self.up1(x5, x4)
        # print("up1:", x.shape)

        x = self.up2(x, x3)
        # print("up2:", x.shape)

        x = self.up3(x, x2)
        # print("up3:", x.shape)

        x = self.up4(x, x1)
        # print("up4:", x.shape)

        x = self.outc(x)
        # print("outc:", x.shape)

        return x.squeeze(1)  # [B, 84]

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size(-1) - x1.size(-1)
        x1 = nn.functional.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




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
    global starttime
    global endtime
    global runtime
    # global direction
    myparams1 = params
    myparams2 = params2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=1).to(device)
    # model.load_state_dict(torch.load("/home/czy/桌面/magx-main1/Codes/optimization/0609Unet.pth", map_location=torch.device('cpu')))  # 加载权重
    model.load_state_dict(torch.load("/home/czy/桌面/magx-main1/Codes/optimization/0418pretrained_encoder_v2.pth", map_location=torch.device('cpu')))  # 加载权重
    model.to(device)
    model.eval()



    tracking_model = MLP().to(device)
    # tracking_model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0130_differential_mlp_model_v3.pth',map_location=torch.device('cpu')))
    tracking_model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/0130_differential_mlp_model_v3.pth',map_location=torch.device('cpu')))

    # Set the model to evaluation mode10
    tracking_model.eval()
    while True:
        if not worklist.empty():
           
                datai = worklist.get()
                
                datai = datai.reshape(-1, 3)

                starttime = time.time() 
                # 初始化一个列表用于存储差分数据
                data_diff = []

                # 遍历所有传感器对
                for i in range(len(datai)):
                    for j in range(i + 1, len(datai)):
                        # 计算每对传感器的差分数据 (Bx, By, Bz)
                        # diff = datai[i]*(0.46/0.23) - datai[j]*(0.46/0.23)
                        # diff = datai[i]*(0.46/0.7) - datai[j]*(0.46/0.7)
                        diff = datai[i]- datai[j]
                        data_diff.append(diff)

                # 将差分数据转换为 numpy 数组
                data_diff = np.array(data_diff)

               

                # datai = datai.T.reshape(-1)
               
            #     model = MLP().to(device)
            # #     model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/full_data_mlp_model.pth', map_location=torch.device('cpu')))
            #     model.load_state_dict(torch.load('/home/czy/桌面/magx-main1/Codes/optimization/1124full_data_mlp_model_v1.pth', map_location=torch.device('cpu')))
            #     model.eval()


                datai_tensor = torch.FloatTensor(data_diff.reshape(-1)).to(device)

                # 添加批次维度，如果 datai_tensor 是一维的
                if datai_tensor.ndimension() == 1:
                    datai_tensor = datai_tensor.unsqueeze(0)  # 形状变为 (1, num_features)
                print("Input shape:", datai_tensor.shape)
                # 使用模型进行预测
                with torch.no_grad():
                    datai_tensor = torch.FloatTensor(data_diff.reshape(-1)).to(device)  # shape: [84]
                    datai_tensor = datai_tensor.unsqueeze(0).unsqueeze(1)  # shape: [1, 1, 84]
                    reconstruc_tensor = model(datai_tensor)
                    y_pred_tensor = tracking_model(reconstruc_tensor)
                    resulti = y_pred_tensor.cpu().numpy()

                print("Prediction result:", resulti)
                endtime = time.time() 
                runtime = endtime-starttime
                print("runtime",runtime)
                result = [resulti[0][0],
                          resulti[0][1], resulti[0][2]]
                results.put(result)
                current = [datetime.datetime.now()]
                current.append(resulti[0][0])
                current.append(resulti[0][1])
                current.append(resulti[0][2])
                current.append(runtime)
                resultslist.append(current)
                # print(resultslist)
               
                if(len(resultslist)==300):
                     print("Output csv")
                     test = pd.DataFrame(columns=name,data=resultslist)
                    #  test.to_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0605data_Unet_"+str(countnum)+".csv")
                     print("Exited")
                     countnum=countnum+1
                     resultslist=[]
                # current = [datetime.datetime.now()]
                # direction=np.array([np.sin(ang_convert(resulti[7]))*np.cos(ang_convert(resulti[8])),
                #           np.sin(ang_convert(resulti[7]))*np.sin(ang_convert(resulti[8])), np.cos(ang_convert(resulti[7]))])
                # current.append(resulti[0]*1e6)
                # current.append(resulti[1]*1e6)
                # current.append(resulti[2]*1e6)
                # current.append(resulti[4] * 1e2)
                # current.append(resulti[5] * 1e2)
                # current.append(resulti[6] * 1e2)
                # current.append(ang_convert1(resulti[7]))
                # current.append(ang_convert(resulti[8]))
                # # current.append((resulti[7]))
                # # current.append((resulti[8]))
                # resultslist.append(current)
                # print(resultslist)
                # if(len(resultslist)==1000):
                #      print("Output csv")
                #      test = pd.DataFrame(columns=name,data=resultslist)
                #      test.to_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/1116LMmove_"+str(countnum)+".csv")
                #      print("Exited")
                #      countnum=countnum+1
                #      resultslist=[]
                # resultlist.append(current)
                # print(resulti[3])
                # print("Orientation: {:.2f}, {:.2f}, m={:.2f}".format(
                #     resulti[7] / np.pi * 180,
                #     resulti[8] / np.pi * 180 % 360,
                #     np.exp(resulti[3])))
                # print("Position: {:.2f}, {:.2f}, {:.2f}, m={:.2f}, dis={:.2f},orientation: {:.2f},{:.2f}".format(
                #     result[0],
                #     result[1],
                #     result[2],
                #     resulti[3],
                #     np.sqrt(
                #         result[0] ** 2 + result[1] ** 2 + result[2] ** 2),
                #     ang_convert1(resulti[7]),
                #     ang_convert(resulti[8]))),
                    
                
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
    fig = plt.figure(figsize=(8, 8))  # 增加 figsize
    # fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')

    # TODO: add title
    ax.set_xlabel('x(cm)')
    ax.set_ylabel('y(cm)')
    ax.set_zlabel('z(cm)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([0, -15])
    ax.set_zlim([-5, 5])
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
    #     test.to_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/1116LMmovesensor_"+str(countnum1)+".csv")
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
        # still_data = Reading_Data("/home/czy/桌面/magx-main1/1128G1.csv")
        
        # print(offset)
        # offset = np.mean()
        [offset, scale] = calibration.cali_result()
        if not os.path.exists('result'):
            os.makedirs('result')
        # offset = np.mean(still_data.raw_readings, axis=0)
        print("This is the offset")
        print(offset)
        print("This is the offset")
        np.savez('result/calibration.npz', offset=offset, scale=scale)
        # print(np.mean(scale))

    asyncio.run(main(1))  # For tracking 1 magnet
    # asyncio.run(main(2)) # For tracking 2 magnet

