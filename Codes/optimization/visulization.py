import numpy as np
from src.preprocess import Calibrate_Data,Calibrate_Data1
from src.visualize import show_track_1mag_csv_cpp, show_track_2mag_csv_cpp,show_track_1mag_csv_cpp_modify
from config import pSensor_large_smt, pSensor_ear_smt,pSensor_selfcare
if __name__ == "__main__":
    # LM_path = '/home/czy/windows_disk/Users/26911/Documents/linux/groundtruth/czy_0930_bench_3.csv'
    # # Reading_path = '/home/czy/桌面/magbrushdata/calib_1116_front_8.csv'
    Reading_path = '/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0930sensor_z1.csv'
    # cali_path = '/home/czy/桌面/magbrushdata/calib_1116_base_1.csv'
    cali_path ='/home/czy/桌面/magx-main1/07042326_changsha_hotel_Tableside_3.csv'
    cali_path = '/home/czy/windows_disk/Users/26911/Documents/linux/calibrationdata/debug.csv'
    cali_path = '/home/czy/桌面/calibartion_free/smartphone_data/calibration_joint1_07050134_changsha_hotel_tableside_1.csv'
    # cali_path = '/home/czy/桌面/magx-main1/mobilitycali/1201_domitory_room_cali.csv'
    # cali_path = '/home/czy/桌面/read_refdata/1212platform2_v1.csv'
    # cali_path = '/home/czy/桌面/magx-main1/pilot_exp/cali.csv'
    cali_path = '/home/czy/桌面/magx-main1/0709cali1.csv'
    cali_data = Calibrate_Data(cali_path)
    cali_data.show_cali_result()
    # /storage/emulated/0/Documents/debug.csv
    # To track 2 magnets, change the function to show_track_2mag_csv_cpp
   
    # show_track_1mag_csv_cpp(Reading_path, cali_path, LM_path, pSensor_selfcare, 1, False)




