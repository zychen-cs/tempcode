import asyncio
from bleak import BleakClient

DEVICE_ADDRESS = "DD:B4:41:07:9E:FE"  # 改成你的设备 MAC 地址
RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # 发送给设备
TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # 设备发回数据

def handle_rx(_, data):
    print("Received:", data)
    if data == b'PING':
        print("Received PING")
        asyncio.create_task(send_pong())

async def send_pong():
    await client.write_gatt_char(RX_UUID, b'P')  # 回发'P'即可用于RTT测量

async def main():
    global client
    client = BleakClient(DEVICE_ADDRESS)
    await client.connect()
    await client.start_notify(TX_UUID, handle_rx)
    print("Waiting for PING...")
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
