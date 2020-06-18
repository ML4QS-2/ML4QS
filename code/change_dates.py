import datetime
import math

import pandas as pd


def calc_min_sec_etc(second):
    hr_min_sec = datetime.timedelta(seconds=max(second, 0))
    now = datetime.datetime(2020, 6, 16, 17, 0, 0, 0) + hr_min_sec
    now = pd.Timestamp(now).value
    return now


file_names = ['Magnetometer', 'Accelerometer', 'Gyroscope', 'Labels']
col_rename = {
    'Magnetometer': ['Time (s)', 'X', 'Y', 'Z', 'Time'],
    'Accelerometer': ['Time (s)', 'X', 'Y', 'Z', 'Time'],
    'Gyroscope': ['Time (s)', 'X', 'Y', 'Z', 'Time'],
    'Labels': ['Time (s)', 'label', 'Step', 'Duration', 'StepDuration', 'label_start', 'label_end']
}

for file_name in file_names:
    df = pd.read_csv('input_data/' + file_name + '.csv', skipinitialspace=True)

    df['new_timestamp'] = df.apply(lambda row: calc_min_sec_etc(row['Time (s)']), axis=1)

    if file_name == 'Labels':
        df['label_end'] = pd.to_datetime(df['new_timestamp']) + pd.to_timedelta(1, 's')
        df['label_end'] = df['label_end'].astype('int64')

    df.columns = col_rename[file_name]
    print('saving file', file_name)

    df.to_csv('./clean_dataset/dance/' + file_name + '.csv')
