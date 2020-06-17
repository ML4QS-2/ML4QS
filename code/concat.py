import os
import pandas as pd
import math

INPUT_FOLDER = './input_data'

all_dirs = [d for d in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, d))]


data = {}
for d in all_dirs:
    files = os.listdir(os.path.join(INPUT_FOLDER, d))
    
    # Get Max Duration Of Measurement
    max_duration = []
    for f in files:
        if f not in data.keys():
            df = pd.read_csv(os.path.join(INPUT_FOLDER, d, f))
            data[f] = pd.DataFrame(data=None, columns=df.columns)
        cdf = data[f]

        min_time = cdf['Time (s)'].min()
        max_time = cdf['Time (s)'].max()
        if math.isnan(min_time):
            min_time = 0
        if math.isnan(max_time):
            max_time = 0
        
        max_duration.append(max_time)
    
    for f in files:
        df = pd.read_csv(os.path.join(INPUT_FOLDER, d, f))
        print(d, f, max_duration)
        if f == 'Labels.csv':
            df['step'] = df['Duration'].apply(lambda x: range(x))
            df = df.explode('step')
            df['Time (s)'] = df['Time (s)'] + df['step']
            print(df)

        df['Time (s)'] = df['Time (s)'] + max(max_duration)
        data[f] = data[f].append(df)

for f, df in data.items():
    print(f)
    df.to_csv(os.path.join(INPUT_FOLDER, f), index=False)
