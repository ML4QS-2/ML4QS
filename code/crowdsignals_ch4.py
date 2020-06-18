##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import copy
import sys
from pathlib import Path

import pandas as pd
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import NumericalAbstraction
from util.VisualizeDataset import VisualizeDataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = sys.argv[1] if len(sys.argv) > 1 else 'chapter3_result_final.csv'
RESULT_FNAME = sys.argv[2] if len(sys.argv) > 2 else 'chapter4_result.csv'

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = pd.to_datetime(dataset.index)

# Let us create our visualization class again.
DataViz = VisualizeDataset(__file__)

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

# Chapter 4: Identifying aggregate attributes.

# First we focus on the time domain.

# OLD: Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
# NEW: Set the window sizes to the number of instances representing 1 second, 8 seconds, 16 seconds  and 24 seconds
window_sizes = [int(float(1000) / milliseconds_per_instance), int(float(4000) / milliseconds_per_instance),
                int(float(8000) / milliseconds_per_instance)]

NumAbs = NumericalAbstraction()
dataset_copy = copy.deepcopy(dataset)
# for ws in window_sizes:
#     dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_phone_X'], ws, 'mean')
#     dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_phone_X'], ws, 'std')
#
# DataViz.plot_dataset(dataset_copy, ['acc_phone_X', 'acc_phone_X_temp_mean', 'acc_phone_X_temp_std', 'label'],
#                      ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])

# ws = int(float(0.5 * 60000) / milliseconds_per_instance) $ FIXME they only use one window size! We do multiple
# ws = int(float(8000) / milliseconds_per_instance)
ws = int(float(4000) / milliseconds_per_instance)
# for ws in window_sizes: #FIXME uncomment for multiple ws's
selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')

# DataViz.plot_dataset(dataset,
#                      ['acc_phone_X', 'gyr_phone_X', 'mag_phone_X', 'pca_1', 'label'],
#                      ['like', 'like', 'like', 'like', 'like'],
#                      ['line', 'line', 'line', 'line', 'points'])
#
# # support for labels is useless in our case
# CatAbs = CategoricalAbstraction()
# dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03,
#                                       int(float(8000) / milliseconds_per_instance), 2)
#
# Now we move to the frequency domain, with the same window size.

FreqAbs = FourierTransformation()
fs = float(1000) / milliseconds_per_instance  # Todo change?

periodic_predictor_cols = ['acc_phone_X', 'acc_phone_Y', 'acc_phone_Z',
                           'gyr_phone_X', 'gyr_phone_Y', 'gyr_phone_Z',
                           'mag_phone_X', 'mag_phone_Y', 'mag_phone_Z']
#
# data_table = FreqAbs.abstract_frequency(copy.deepcopy(dataset), ['acc_phone_Y'],
#                                         int(float(4000) / milliseconds_per_instance), fs)

# Spectral analysis.

# DataViz.plot_dataset(data_table, ['acc_phone_Y_max_freq', 'acc_phone_Y_freq_weighted', 'acc_phone_Y_pse', 'label'],
#                      ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
# we use 4s
ws_freq = int(float(4000) / milliseconds_per_instance)
dataset = FreqAbs.abstract_frequency(dataset, periodic_predictor_cols, ws_freq, fs)

# Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

ws = int(float(4000) / milliseconds_per_instance)  # we remove 10% of the data for every second.
# The percentage of overlap we allow
window_overlap = 0.9
skip_points = int((1 - window_overlap) * ws)
dataset = dataset.iloc[::skip_points, :]

dataset.to_csv(DATA_PATH / RESULT_FNAME)

DataViz.plot_dataset(dataset,
                     ['acc_phone_X', 'gyr_phone_X', 'mag_phone_X', 'pca_1', 'label'],
                     ['like', 'like', 'like', 'like', 'like'],
                     ['line', 'line', 'line', 'line', 'points'])
