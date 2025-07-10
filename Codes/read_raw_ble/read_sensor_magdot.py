import asyncio
import struct
import datetime
import numpy as np
from bleak import BleakClient
import pandas as pd
from pynput import mouse
import atexit
# Nordic NUS characteristic for RX, which should be writable`
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# Nordic NUS characteristic for TX, which should be readable
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

sensors = np.zeros((11, 3))
result = []
name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3',
        'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 9', 'Sensor 10']

stop_program = False
ctime = "07052338"
place = "changsha_hotel"
scene = "tableside"
order = 1

@atexit.register
def clean():
    print("Output csv")
    test = pd.DataFrame(columns=name, data=result)
    # test.to_csv("{}_{}_{}_{}.csv".format(ctime, place, scene, order))
    test.to_csv("0715_1.csv")
    print("Exited")

def on_click(x, y, button, pressed):
    global stop_program
    if pressed and button == mouse.Button.right:  # Only react to the right mouse button
        print(f"Mouse clicked at ({x}, {y}) with {button}")
        stop_program = True
        return False

def notification_handler(sender, data):
    num = 10
    global sensors
    global result
    current = [datetime.datetime.now()]
    for i in range(num):
        sensors[i, 0] = struct.unpack('f', data[16 * i: 16 * i + 4])[0]
        sensors[i, 1] = struct.unpack('f', data[16 * i + 4: 16 * i + 8])[0]
        sensors[i, 2] = struct.unpack('f', data[16 * i + 8: 16 * i + 12])[0]
        print(f"Sensor {i+1}: {sensors[i, 0]}, {sensors[i, 1]}, {sensors[i, 2]}")
        current.append(f"({sensors[i, 0]}, {sensors[i, 1]}, {sensors[i, 2]})")
    print("############")
    result.append(current)

async def run(address, loop):
    async with BleakClient(address, loop=loop) as client:
        x = await client.is_connected()
        print(f"Connected: {x}")
        print("Press Enter to quit...")
        await client.start_notify(UART_TX_UUID, notification_handler)
        while not stop_program:
            await asyncio.sleep(0.01)
        await client.stop_notify(UART_TX_UUID)
        print("Stopped notification")

async def main():
    global address
    loop = asyncio.get_event_loop()
    listener = mouse.Listener(on_click=on_click)
    listener.start()
    try:
        await run(address, loop)
    finally:
        listener.stop()

if __name__ == '__main__':
    address = "CE:E1:85:67:A6:2B"
    # address = "E8:A8:DA:7C:8D:77"
    asyncio.run(main())
