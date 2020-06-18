##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from util.VisualizeDataset import VisualizeDataset


def main():
    # Set up file names and locations.
    DATA_PATH = Path('./intermediate_datafiles/')
    DATASET_FNAME = sys.argv[1] if len(sys.argv) > 1 else 'chapter2_result.csv'
    RESULT_FNAME = sys.argv[2] if len(sys.argv) > 2 else 'chapter3_result_outliers.csv'

    # Next, import the data from the specified location and parse the date index.
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)

    except IOError as e:
        print('File not found, try to run the preceding crowdsignals scripts first!')
        raise e

    # We'll create an instance of our visualization class to plot the results.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance using the first two rows.
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

    # Step 1: Let us see whether we have some outliers we would prefer to remove.

    # Determine the columns we want to experiment on.
    outlier_columns = ['acc_phone_X', 'acc_phone_Y', 'acc_phone_Z',
                       'gyr_phone_X', 'gyr_phone_Y', 'gyr_phone_Z',
                       'mag_phone_X', 'mag_phone_Y', 'mag_phone_Z', ]

    # Create the outlier classes.
    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()

    # And investigate the approaches for all relevant attributes.
    for col in outlier_columns:

        print(f"Applying outlier criteria for column {col}")

        # And try out all different approaches. Note that we have done some optimization
        # of the parameter values for each of the approaches by visual inspection.
        dataset = OutlierDistr.chauvenet(dataset, col)
        DataViz.plot_binary_outliers(dataset, col, col + '_outlier')
        dataset = OutlierDistr.mixture_model(dataset, col)
        DataViz.plot_dataset(dataset, [col, col + '_mixture'], ['exact', 'exact'], ['line', 'points'])
        # This requires:
        # n_data_points * n_data_points * point_size =
        # 31839 * 31839 * 32 bits = ~4GB available memory

        try:
            dataset = OutlierDist.simple_distance_based(dataset, [col], 'euclidean', 0.10, 0.99)
            DataViz.plot_binary_outliers(dataset, col, 'simple_dist_outlier')
        except MemoryError as e:
            print('Not enough memory available for simple distance-based outlier detection...')
            print('Skipping.')

        try:
            dataset = OutlierDist.local_outlier_factor(dataset, [col], 'euclidean', 5)
            DataViz.plot_dataset(dataset, [col, 'lof'], ['exact', 'exact'], ['line', 'points'])
            DataViz.plot_dataset_boxplot(dataset, ['lof'])
            # print(col, dataset['lof'].describe())
            qtls = list(dataset['lof'].quantile([0.01, 0.25, 0.5, 0.75, 0.99]))
            # print(col, qtls)
            #print(col, qtls[4])
            
            dataset['lof_outliers'] = False
            dataset.loc[(dataset['lof'] > qtls[4]), 'lof_outliers'] = True

            DataViz.plot_binary_outliers(dataset, col, 'lof_outliers')
        except MemoryError as e:
            print('Not enough memory available for lof...')
            print('Skipping.')

        # Remove all the stuff from the dataset again.
        cols_to_remove = [col + '_outlier', col + '_mixture', 'simple_dist_outlier', 'lof', 'lof_outliers']
        for to_remove in cols_to_remove:
            if to_remove in dataset:
                del dataset[to_remove]

    # We take Chauvenet's criterion and apply it to all but the label data...

    for col in [c for c in dataset.columns if not 'label' in c]:
        print(f'Measurement is now: {col}')
        dataset = OutlierDistr.chauvenet(dataset, col)
        dataset.loc[dataset[f'{col}_outlier'] == True, col] = np.nan
        del dataset[col + '_outlier']

    dataset.to_csv(DATA_PATH / RESULT_FNAME)


if __name__ == '__main__':
    main()
