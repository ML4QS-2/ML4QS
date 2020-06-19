##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

from pathlib import Path

import pandas as pd
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.LearningAlgorithms import ClassificationAlgorithms, plotcv
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from util import util
from util.VisualizeDataset import VisualizeDataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter5_result.csv'
DATASET_NOVELFNAME = 'chapter5-novel-data_result.csv'
RESULT_FNAME = 'chapter7_nove-classification_result.csv'
EXPORT_TREE_PATH = Path('./figures/crowdsignals_ch7_classification/')


DATASET_NOVELFNAME = DATASET_FNAME


# Next, we declare the parameters we'll use in the algorithms.
N_FORWARD_SELECTION = 50

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
    dataset_novel = pd.read_csv(DATA_PATH / DATASET_NOVELFNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e




dataset.index = pd.to_datetime(dataset.index)

dataset_novel.index = pd.to_datetime(dataset_novel.index)

# Let us create our visualization class again.
DataViz = VisualizeDataset(__file__)

# Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

# We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
# for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
# cases where we do not know the label.

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7,
                                                                               filter=True, temporal=False)

novel_X, novel_n_X, novel_y, novel_n_y = prepare.split_single_dataset_classification(dataset_novel, ['label'], 'like', 0.999,
                                                                               filter=True, temporal=False)

print(novel_X)
print(novel_y)

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))

# Select subsets of the features that we will consider:
basic_features = ['acc_phone_X', 'acc_phone_Y', 'acc_phone_Z',
                  'gyr_phone_X', 'gyr_phone_Y', 'gyr_phone_Z',
                  'mag_phone_X', 'mag_phone_Y', 'mag_phone_Z']
pca_features = ['pca_1', 'pca_2', 'pca_3']
time_features = [name for name in dataset.columns if '_temp_' in name]
freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
print('#basic features: ', len(basic_features))
print('#PCA features: ', len(pca_features))
print('#time features: ', len(time_features))
print('#frequency features: ', len(freq_features))
# cluster_features = ['cluster'] # We didn't include clusters
# print('#cluster features: ', len(cluster_features))
features_after_chapter_3 = list(set().union(basic_features, pca_features))
features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))

# Based on the plot we select the top 10 features (note: slightly different compared to Python 2, we use
# those feartures here).
selected_features = ['gyr_phone_Z_freq_2.0_Hz_ws_16', 'gyr_phone_X_freq_1.0_Hz_ws_16', 'mag_phone_Z_freq_0.75_Hz_ws_16',
                     'mag_phone_Y_freq_0.75_Hz_ws_16', 'pca_1_temp_std_ws_16', 'acc_phone_Z_temp_mean_ws_16',
                     'acc_phone_X_freq_0.0_Hz_ws_16', 'pca_3_temp_std_ws_16', 'mag_phone_X_freq_1.25_Hz_ws_16',
                     'mag_phone_Z_freq_0.25_Hz_ws_16']

learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, selected_features]
feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Selected features']

# And we study two promising ones in more detail. First, let us consider the decision tree, which works best with the
# selected features.
#
class_train_y, class_test_y, class_train_prob_y, class_test_prob_y, gscv = learner.random_forest(
    train_X[selected_features], train_y, test_X[selected_features],
    gridsearch=True, print_model_details=True)



test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)

plotcv(gscv)

best_model = gscv.best_estimator_
print(gscv.best_estimator_)


from sklearn.metrics import accuracy_score, confusion_matrix


y = best_model.predict(novel_X[selected_features])
y_prob = best_model.predict_proba(novel_X[selected_features])
y_frame = pd.DataFrame(y_prob, columns=best_model.classes_)


print(test_y)
print(y_frame)
print(class_test_y)
print(class_train_prob_y.columns)


# print(labels)
confusion_matrix(novel_y, y_frame, labels=best_model.classes_)

# print(accuracy_score(novel_X['label'], y, normalize=False))
#test_cm = eval.confusion_matrix(y, y_prob, y_prob.columns)
#DataViz.plot_confusion_matrix(test_cm, y_prob.columns, normalize=False)

