import asyncio
import struct
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


# Nordic NUS characteristic for RX, which should be writable`
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# Nordic NUS characteristic for TX, which should be readable
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

# sensors = np.zeros((17, 3))
sensors = np.zeros((9, 3))
result = []
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8','Sensor 9', 'Sensor 10','Sensor 11','Sensor 12']
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4','Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8','Sensor 9', 'Sensor 10']
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']
# name = ['Time Stamp', 'Sensor 1']
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8','Sensor 9','Sensor 10']
name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
        'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3']
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8',
#         'Sensor 9', 'Sensor 10','Sensor 11','Sensor 12',
#         'Sensor 13', 'Sensor 14','Sensor 15','Sensor 16']


start_offset = None  # 用于第一次接收时初始化参考偏移
offsetnum=[]
lattency=[]
offset_list=[]
flag=False
idnum =0
@atexit.register
def clean():
    print("Output csv")
    test = pd.DataFrame(columns=name, data=result)
    # test.to_csv("/home/czy/桌面/magx-main1/0514_cph_mag2_name_2.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/0330sensor21.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/newregion3/area4/3cm/noise18.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/pilot_exp/Scissorsdata.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/pilot_exp/Scissorsdata.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/noise_type2_newdir/3cm/noise108.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/noisetest/type4/3cm/noise36.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/noisedataset/3cm/pos9_interference18.csv")
    test.to_csv("/home/czy/桌面/magx-main1/0709cali1.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/type3_pos12.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/0322data_test2.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/calinet1/-3cm_x0.5y0.5/bench_up/0219_I2C_8sensor_441_9_6.csv")
    # test.to_csv("/home/czy/桌面/magx-main1/calinet1/-3cm_x0.5y0.5/bench_up/0219_I2C_8sensor_441_G.csv")
    # test.to_csv("{}_{}_{}_1.csv".format(ctime,place, scene))
    print("Exited")



def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    # num = 16
    num = 8 
    # num = 3 
    global sensors
    global result
    global start_offset
    global idnum
    global flag
    global offset_list
    current = [datetime.datetime.now()]
    for i in range(0,num):
        sensors[i, 0] =  struct.unpack('f', data[12 * i: 12 * i + 4])[0]
        sensors[i, 1] =  struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
        sensors[i, 2] =  struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
        print("Sensor " + str(i+1)+": " +
              str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2]))
        current.append(
            "("+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2])+")")
    # send_time = struct.unpack('<I', data[-4:])[0]  # 发送端 micros() 时间戳（单位 us）
    # recv_time = time.perf_counter() * 1_000_000    # 接收端当前时间（单位 us）

    # 初始化 offset（只记录前 50 个 offset）
    # if len(offset_list) < 50:
    #     offset = recv_time - send_time
    #     offset_list.append(offset)
    #     print(f"[初始化 offset 收集中] 当前 offset = {offset:.0f} us")
    #     return  # ✅ 替代 continue，退出本轮处理

    # # 使用最小 offset，提高精度
    # start_offset = min(offset_list)
    # expected_send_arrival = start_offset + send_time
    # latency = recv_time - expected_send_arrival
    # lattency.append(latency)
    # print(f"BLE传输时延: {latency:.0f} us")

    # # 打印平均时延
    # if len(lattency) >= 100:
    #     print("=================")
    #     print(f"BLE 单向传输平均时延: {np.mean(lattency):.0f} us")
    # for i in range(0,4):
    #     sensors[i, 1] = -struct.unpack('f', data[12 * i: 12 * i + 4])[0]
    #     sensors[i, 2] = -struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
    #     sensors[i, 0] =  struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
    #     print("Sensor " + str(i+1)+": " +
    #           str(sensors[i, 2]) + ", " + str(sensors[i, 0]) + ", " + str(sensors[i, 1]))
    #     current.append(
    #         "("+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2])+")")
    # for i in range(4,8):
    #     sensors[i, 1] =  struct.unpack('f', data[12 * i: 12 * i + 4])[0]
    #     sensors[i, 2] = -struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
    #     sensors[i, 0] = -struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
    #     print("Sensor " + str(i+1)+": " +
    #           str(sensors[i, 2]) + ", " + str(sensors[i, 0]) + ", " + str(sensors[i, 1]))
    #     current.append(
    #         "("+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2])+")")
    #battery_voltage = struct.unpack('f', data[12 * num: 12 * num + 4])[0]
    #print("Battery voltage: " + str(battery_voltage))
    print("############")
    result.append(current)




async def run(address, loop):
    async with BleakClient(address, loop=loop) as client:
        # wait for BLE client to be connected
        x = await client.is_connected()
        print("Connected: {0}".format(x))
        print("Press Enter to quit...")
        # wait for data to be sent from client
        await client.start_notify(UART_TX_UUID, notification_handler)
        while True:
            await asyncio.sleep(0.01)

async def main():
    global address
    await asyncio.gather(
        asyncio.create_task(run(address, asyncio.get_event_loop()))
    )
if __name__ == '__main__':
    # address = ("D0:2D:B2:31:91:88")
    # address = ("FA:0A:21:CD:68:61")
    # address = ("E8:A8:DA:7C:8D:77")
    # address = ("CE:E1:85:67:A6:2B")
    # address = ("DE:31:41:0F:0F:0D")
    # C7:E4:C1:F3:87:BD
    # EE:66:70:D4:74:5D
    # DE:31:41:0F:0F:0D
    # address = ("F2:50:16:40:B3:D5")
    # address = ("D7:48:48:E9:A8:0D")
    # address = ("D5:30:3B:09:73:CF")
    # address = ("F0:33:85:3D:67:9D")


    # address =("E0:AC:47:4E:97:F7")

    # address = ("CD:7A:5F:1E:8B:07")

    # address = ("EA:AC:53:27:34:33")

    # address = ("E3:B2:FC:BD:2B:1E")

    address = ("DD:B4:41:07:9E:FE")
    # address = ("E0:F6:E5:55:9F:E3")
    asyncio.run(main())
