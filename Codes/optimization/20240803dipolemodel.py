from os import TMP_MAX
import numpy as np
from numpy.core.fromnumeric import size
from multiprocessing import Pool
import datetime
from tqdm import tqdm
from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform
import datetime
import os
from src.solver import Solver_jac
from src.simulation import expression, Simu_Data
import cppsolver as cs
import copy
import pandas as pd
data = pd.read_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0804test_2.csv")

Xs = 0
Ys = 0
Zs = 0
pSensor = 1e-2 * np.array([
    
    [ 0, 0, 0]
   
    
])
result=[]
name = ["sensor5"]
for i in range(0,len(data)):
    x = data["x"][i]
    y = data["y"][i]
    z = data["z"][i]
    # x=-0.3
    # y=-5.5
    # z=2.3
    theta = data["theta"][i]
    # theta = np.pi - theta
    phi = data["phy"][i]
    # phi = (phi + np.pi) % (2 * np.pi)
    # print(np.sqrt(x**2+y**2+z**2)*np.sin(theta)*np.cos(phi))
    # params = np.array([50 / np.sqrt(2) * 1e-6, 50 / np.sqrt(2) * 1e-6, 0, np.log(
    #                     0.46), 1e-2 *x, 1e-2 * y, 1e-2 * z, theta, phi])
    #                 # print(np.linalg.norm(np.array(params[4:7]), ord=2))
    # B = cs.calB(pSensor.reshape(-1), params.reshape(-1))
    # print(B)
    VecM = np.array([np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)]) * 1e-7 * np.exp(np.log(0.46))
    VecR = np.array([(x*0.01)-Xs, (y*0.01)-Ys, (z*0.01)-Zs])
    
# 计算 VecR 的范数
    NormR = np.linalg.norm(VecR)
    B = (3.0 * VecR * (np.dot(VecM, VecR)) / NormR**5 - VecM / NormR**3)
    # scalar_part = 3.0 * VecR  (np.dot(VecM, VecR) / NormR**5)
    # B = scalar_part - VecM / NormR**3
    
    B_x = B[0]*1000000
    B_y = B[1]*1000000
    B_z = B[2]*1000000
   
    current=[]
    current.append(
            "("+str(B_x) + ", " + str(B_y) + ", " + str(B_z)+")")
    print(current)
    result.append(current)
test = pd.DataFrame(columns=name,data=result)
# test.to_csv("/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0803compare_sensor5.csv")
print("Exited")