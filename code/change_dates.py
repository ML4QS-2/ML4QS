import datetime
import math

import pandas as pd


def calc_min_sec_etc(second):
    s = float(second)
    sec = math.floor(s)

    m_sec = str(s - int(s))[2:5]

    hr_min_sec = str(datetime.timedelta(seconds=sec))

    h = int(hr_min_sec.split(':')[0])
    m = int(hr_min_sec.split(':')[1])
    s = int(hr_min_sec.split(':')[2])

    # 19 = 17 hours --> GMT?
    now = int(round(datetime.datetime(2020, 6, 16, 19 + h, m, s, int(m_sec)).timestamp() * 1000000000))
    return now


file_names = ['Magnetometer', 'Accelerometer', 'Gyroscope', 'Labels']
col_rename = {
    'Magnetometer': ['Time (s)', 'X', 'Y', 'Z', 'Time'],
    'Accelerometer': ['Time (s)', 'X', 'Y', 'Z', 'Time'],
    'Gyroscope': ['Time (s)', 'X', 'Y', 'Z', 'Time'],
    'Labels': ['Time (s)', 'label', 'Step', 'Duration', 'label_start','label_end']
}

for file_name in file_names:
    df = pd.read_csv('input_data/' + file_name + '.csv', skipinitialspace=True)

    df['new_timestamp'] = df.apply(lambda row: calc_min_sec_etc(row['Time (s)']), axis=1)

    if file_name == 'Labels':
        df['label_end'] = pd.to_datetime(df['new_timestamp']) + pd.to_timedelta(df['Duration'], 's')
        df['label_end'] = df['label_end'].astype('int64')

    df.columns = col_rename[file_name]
    print('saving file', file_name)

    df.to_csv('./clean_dataset/dance/' + file_name + '.csv')
