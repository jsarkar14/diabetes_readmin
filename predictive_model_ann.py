import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from data_structuring import read_filter_clean_map_csv

# Reading in the main dataset calling the function defined in the other file
df = read_filter_clean_map_csv()

column_names = df.columns.values.tolist()

# Determining the total number of rows and columns
df_nrows, df_ncols = df.shape

# This code is not super-generalized so it assumes that the user knows that there are 3 outcomes
# as defined in the other file (for more documentation check mapping_readmittance() in the other file
Positive_indices = df[df['readmitted'] == 1].index.tolist()
Zero_indices = df[df['readmitted'] == 0].index.tolist()
Negative_indices = df[df['readmitted'] == -1].index.tolist()

# We are using a 50/50 split for train and split
train_test_ratio_positive = 0.4

positive_zero_ratio = np.float(len(Positive_indices)) / np.float(len(Zero_indices))
positive_negative_ratio = np.float(len(Positive_indices)) / np.float(len(Negative_indices))

train_test_ratio_zero = train_test_ratio_positive*positive_zero_ratio
train_test_ratio_negative = train_test_ratio_positive*positive_negative_ratio

N_cross_validations = 1

# Repeating the whole model train/test N_cross_validation times to understand if the model is over-fitted
for index in xrange(N_cross_validations):

    # Dividing the dataset into a training and test set
    # Sampling is done separately for sampling adjusted to the rate of occurrence of actual outcomes
    list1, list2 = train_test_split(Positive_indices, test_size=1.0 - train_test_ratio_positive)
    list3, list4 = train_test_split(Zero_indices, test_size=1.-train_test_ratio_zero)
    list5, list6 = train_test_split(Negative_indices, test_size=1.-train_test_ratio_negative)

    # the 3 lists are now joined to form a training and test set indices and the indices are shuffled
    training_indices = np.random.permutation(np.append(np.append(list1, list3), list5))
    testing_indices = np.random.permutation(np.append(np.append(list2, list4), list6))

    # Using the list of indices to create the actual training and test sets
    Training_data_set = df.loc[training_indices]
    Testing_data_set = df.loc[testing_indices]

    # Separating out the input (X) and output(Y); assuming that the output is the last column
    cols = df.columns.values.tolist()
    X_training = Training_data_set[cols[:df_ncols-1]]
    Y_training = Training_data_set[cols[-1]]

    # Creating a random forest classifier and fitting it to the training set
    MLP = MLPClassifier(solver='lbfgs',alpha=1.0e-4,
                        hidden_layer_sizes=(10,2),
                        activation='logistic',
                        validation_fraction=0.25,
                        max_iter=500
                        )
    MLP.fit(X_training,Y_training)
    print MLP.n_iter_

    # Using the test set to determine the score
    X_test = Testing_data_set[cols[:df_ncols-1]]
    Y_test = Testing_data_set[cols[-1]]

    # Predict class and class probabilities for X_test
    rf_probs = MLP.predict_proba(X_test)
    Y_pred = MLP.predict(X_test)

    # Returns the mean accuracy on the given test data and labels.
    # In multi-label classification, this is the subset accuracy which is a harsh metric
    # since you require for each sample that each label set be correctly predicted.
    score = log_loss(Y_test, rf_probs)

    print 'Score for the',index + 1,'-th iteration is =',score
    print confusion_matrix(Y_test, Y_pred)

    # x = np.argsort(RF.feature_importances_)
    # print 'Decreasing oder of Importance of features in prediction of 30 day readmission probability', '\n', [column_names[i] for i in x]

    # for i in xrange(100):
    #     print Y_test.iloc[i],rf_probs[i,:]
