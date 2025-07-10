# The user has uploaded a CSV file and requested to read the data from it,
# specifically splitting the data in parentheses. Let's first read the file
# and then process it accordingly.

# Load the CSV file
import pandas as pd



# Here is the complete code including the imports, reading the CSV file, and processing the data to extract the sensor values.

import pandas as pd
import numpy as np
import re

# Function to extract float values from a string within parentheses
def extract_values(s):
    return list(map(float, re.findall(r'\(([^()]+)\)', s)[0].split(', ')))

# Function to process a single row of sensor data
def process_row(row):
    row_sensor_values = []
    for sensor_data in row:
        sensor_values = extract_values(sensor_data)
        row_sensor_values.extend(sensor_values)
    return row_sensor_values

# Read the CSV file containing the sensor data
csv_file_path = '/home/czy/windows_disk/Users/26911/Documents/linux/trackingdata/0114sensor1_1.csv'
sensor_data_df = pd.read_csv(csv_file_path)

# Apply the process_row function to each row of sensor data
# Excluding the first two columns which are 'Unnamed: 0' and 'Time Stamp'
sensor_columns = sensor_data_df.columns[2:]
sensor_data_processed = sensor_data_df[sensor_columns].apply(process_row, axis=1)

# Convert the series of lists into a 2D numpy array
sensor_data_np = np.array(sensor_data_processed.tolist())
print(sensor_data_np[0])
# # Display the shape of the numpy array to confirm its dimensions
# sensor_data_np.shape

# # The sensor_data_np array now contains the processed sensor data in the desired format
# # Below are the first row of the array and its shape as confirmation
# (sensor_data_np[0], sensor_data_np.shape)
