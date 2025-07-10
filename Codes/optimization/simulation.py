# from src.simulation import Simu_Test, simulate_2mag_3type, Simu_with_magnet
from src.simulation import Simu_Test, simulate_2mag_3type
import numpy as np
from config import pSensor_smt
from multiprocessing import Pool
from config import pSensor_large_smt, pSensor_small_smt, pSensor_median_smt
import os
import matplotlib.pyplot as plt
if __name__ == "__main__":

    pSensor_test = 1e-2 * np.array([
        # [4.9, -4.9, -1.63],
        # [-4.9, -4.9, -1.63],
        # [4.9, 4.9, -1.63],
        # [-4.9, 4.9, -1.63],
        # [0, 4.9, 1.63],
        # [4.9, 0, 1.63],
        # [0, -4.9, 1.63],
        # [-4.9, 0, 1.63],
         [2.2606,  0,      -8],
        [0,   -2.5527,    -8],
        [-2.5527, -0.9779,      -8],
        [-0.5461,5.7531,   -8],
        [-2.5527, 4.5212,      -8],
        [-2.5527,0.6985,   -8],
        [2.2606,  0,      8],
        [0,   -2.5527,    8],
        [-2.5527, -0.9779,      8],
        [-0.5461,5.7531,   8],
        [-2.5527, 4.5212,      8],
        [-2.5527,0.6985,   8],
    ])
    testmodel = Simu_Test(10, 30, [4, 5, 6, 7, 8], resolution=300)
    testmodel.compare_3_noise(2)
