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


file_names = ['Magnetometer', 'Accelerometer', 'Gyroscope']  # Labels

for file_name in file_names:
    df = pd.read_csv('datasets/dancing-raw/' + file_name + '.csv', skipinitialspace=True)

    df['new_timestamp'] = df.apply(lambda row: calc_min_sec_etc(row['Time (s)']), axis=1)

    print('saving file', file_name)

    df.to_csv('datasets/dance/' + file_name + '.csv')
