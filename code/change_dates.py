import datetime
import math

import pandas as pd

file_name = 'Magnetometer'
df = pd.read_csv('datasets/dance/' + file_name + '.csv', skipinitialspace=True)


def calc_min_sec_etc(second):
    s = float(second)
    sec = math.floor(s)

    m_sec = str(s - int(s))[2:5]

    hr_min_sec = str(datetime.timedelta(seconds=sec))

    h = int(hr_min_sec.split(':')[0])
    m = int(hr_min_sec.split(':')[1])
    s = int(hr_min_sec.split(':')[2])

    now = int(round(datetime.datetime(2020, 6, 1, 17 + h, m, s, int(m_sec)).timestamp() * 1000000000))
    return now


df['new_timestamp'] = df.apply(lambda row: calc_min_sec_etc(row.Time), axis=1)

print('saving file')

df.to_csv('datasets/dance/' + file_name + '.csv')
