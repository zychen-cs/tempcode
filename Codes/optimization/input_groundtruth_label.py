import datetime
import atexit
import time
import pandas as pd
import atexit

NUMBER = 2
name = ['Time Stamp', 'Label']
result = []

@atexit.register
def clean():
    print("Output csv")
    test = pd.DataFrame(columns=name, data=result)
    # test.to_csv("timestamp_0310_xsens_far_near2.csv")
    test.to_csv("timestamp_0301_1.csv")
    print("Exited")


while True:
    
    label = input("input:")

    for i in range(NUMBER):
        current = [datetime.datetime.now()]
        # current = [datetime.datetime(2023, 12, 17, 21, 33, 38, 99000)]
        current.append(label)
        result.append(current)
        # current1= [datetime.datetime(2023, 12, 17, 21, 42, 35, 99000)]
        # current1.append(label)
        # result.append(current1)
        time.sleep(0.1)
    