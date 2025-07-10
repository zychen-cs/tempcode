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

sensors = np.zeros((11, 3))
# sensors = np.zeros((9, 3))
result = []
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8','Sensor 9', 'Sensor 10','Sensor 11','Sensor 12']
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8','Sensor 9', 'Sensor 10']
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']
name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
        'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8','Sensor 9','Sensor 10']
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']
# name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
#         'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8',
#         'Sensor 9', 'Sensor 10','Sensor 11','Sensor 12',
#         'Sensor 13', 'Sensor 14','Sensor 15','Sensor 16']



@atexit.register
def clean():
    print("Output csv")
    test = pd.DataFrame(columns=name, data=result)
    # test.to_csv("/home/czy/windows_disk/Users/26911/Documents/linux/calibrationdata/0419_cali_magx_600_1_1.csv")
    test.to_csv("0623_3.csv")
    print("Exited")



def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    # num = 16
    num = 10
    global sensors
    global result
    current = [datetime.datetime.now()]
    for i in range(0,num):
        sensors[i, 0] =  struct.unpack('f', data[12 * i: 12 * i + 4])[0]
        sensors[i, 1] =  struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
        sensors[i, 2] =  struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
        print("Sensor " + str(i+1)+": " +
              str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2]))
        current.append(
            "("+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2])+")")
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
    # address = ("C7:E4:C1:F3:87:BD")
    address = ("DE:31:41:0F:0F:0D")
    # C7:E4:C1:F3:87:BD
    # EE:66:70:D4:74:5D
    # address = ("F2:50:16:40:B3:D5")
    # address = ("CE:E1:85:67:A6:2B")
    # address = ("CC:42:8E:0E:D5:D5")
    # address = ("E0:F6:E5:55:9F:E3")
    asyncio.run(main())
