import os
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.losses import MeanSquaredLogarithmicError
from pandas import get_dummies, concat, read_csv, DataFrame, set_option
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_train_data(fpath):
    """This function import traning data and reformats the columns in desired format
    :param fpath: path of the training dataset
    :return: training dataframe with Item ID
    """
    set_pandas()
    df = read_csv(fpath)
    train_df = df.iloc[:, [1, 2, 3, 4, 5, 6, 8, 7]]
    train_ids = df.iloc[:, [0]]
    print(train_df.shape)
    print(train_df.tail(10))
    return train_df, train_ids


def one_hot_encode(df, colnames):
    """This function performs one-hot encoding of the columns
    :param df: input df
    :param colnames: columns to be one-hot encoded
    :return: dataframe
    """
    for col in colnames:
        oh_df = get_dummies(df[col], prefix=col)
        df = concat([oh_df, df], axis=1)
        df = df.drop([col], axis=1)
    return df


def normalize_columns(df, colnames, scaler):
    """Performs Normalization using MinMaxScaler class in Sckikit-learn"""
    for col in colnames:
        # Create x, where x the 'scores' column's values as floats
        x = df.loc[:, [col]]
        x = df[[col]].values.astype(float)
        # Create a minimum and maximum processor object
        # Create an object to transform the data to fit minmax processor
        x_scaled = scaler.fit_transform(x)
        # Run the normalizer on the dataframe
        df[col] = DataFrame(x_scaled)
    print(f'''Normalized Columns: {colnames} using MinMaxScaler.''')
    return df


def check_unique_value(df, colnames):
    """Gets unique value counts for all selected columns in the dataframe including NaN values and PrettyPrints the
    dicitonary
    :param df:
    :param colnames:
    :return: pretty print a dictionary
    """
    mydict = {}
    for col in colnames:
        val_count = (df[col].value_counts(dropna=False)).to_dict()
        mydict[col] = val_count
    pprint(mydict)
    return


def prepare_data(df):
    """Prepares the Input data for applying machine learning algorithm
    :param df:
    :return: df
    """
    df = df.drop(['Market_Category', 'Date'], axis=1)

    # normalize the columns
    scaler = MinMaxScaler()
    df = normalize_columns(df, ['Demand', 'High_Cap_Price', 'Grade'], scaler)
    df = one_hot_encode(df, ['State_of_Country', 'Product_Category'])
    return df


def set_pandas():
    # Setting display option in pandas
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)
    set_option('display.max_colwidth', -1)


def split_dataset(df, test_size, seed):
    """This function randomly splits (using seed) train data into training set and validation set. The test size
    paramter specifies the ratio of input that must be allocated to the test set
    :param df: one-hot encoded dataset
    :param test_size: ratio of test-train data
    :param seed: random split
    :return: training and validation data
    """
    ncols = np.size(df, 1)
    X = df.iloc[:, range(0, ncols - 1)]
    Y = df.iloc[:, ncols - 1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return x_train, x_test, y_train, y_test


def get_model(input_size, output_size, magic='relu'):
    """This function creates a baseline feedforward neural network with of given input size and output size
        using magic activation function.
    :param input_size: number of columns in x_train
    :param output_size: no of columns in one hpt
    :param magic: activation function
    :return:Sequential model
    """
    mlmodel = Sequential()
    mlmodel.add(Dense(18, input_dim=input_size, activation=magic))
    # mlmodel.add(LeakyReLU(alpha=0.1))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(256, activation=magic))
    mlmodel.add(Dense(256, activation=magic))
    mlmodel.add(Dense(512, activation=magic))
    mlmodel.add(Dense(512, activation=magic))
    mlmodel.add(Dense(1024, activation=magic))
    mlmodel.add(Dense(output_size))

    # Setting optimizer
    msle = MeanSquaredLogarithmicError()
    # mlmodel.compile(loss=msle, optimizer='adam', metrics=['mean_squared_error'])
    mlmodel.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return mlmodel


def get_baseline_model(input_size, output_size, magic='relu', optimizer='adam', loss='mean_squared_error',
                       metrics='mae'):
    """This function creates a baseline feedforward neural network with of given input size and output size
        using magic activation function.
    :param input_size: number of columns in x_train
    :param output_size: no of columns in one hpt
    :param magic: activation function
    :type optimizer: model optimizer
    :param metrics: metric
    :param loss: loss function
    :return:Sequential model
    """
    mlmodel = Sequential()
    mlmodel.add(Dense(18, input_dim=input_size, activation=magic))
    # mlmodel.add(LeakyReLU(alpha=0.1))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(256, activation=magic))
    mlmodel.add(Dense(256, activation=magic))
    mlmodel.add(Dense(512, activation=magic))
    mlmodel.add(Dense(512, activation=magic))
    mlmodel.add(Dense(1024, activation=magic))
    mlmodel.add(Dense(output_size))

    # Setting optimizer

    msle = MeanSquaredLogarithmicError()
    mlmodel.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])
    # mlmodel.compile(loss=msle, optimizer='adam', metrics=['mean_squared_error'])
    # mlmodel.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return mlmodel


def fit_and_evaluate(model, x_train, y_train, batch_size, epochs, x_test, y_test):
    """fits the model created in the create_baseline_model function on x_train, y_train and evaluates the model
    performance on x_test and y_test using the batch size and epochs parameters
    :param model: Sequential model
    :param x_train: training data
    :param y_train: training label
    :param x_test: testing data
    :param y_test: testing label
    :param batch_size: amount of training data (x_train) fed to the model
    :param epochs: number of times the entire dataset is passed through the network
    :return: tuple of validation_accuracy and validation_loss
    """
    now_time = datetime.now()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                        verbose=1)

    # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    finish_time = datetime.now()
    duration = finish_time - now_time
    minutes, seconds = divmod(duration.seconds, 60)
    print(f'Total training time: {minutes}:{seconds}')
    return history


def plt_plot_mse(history):
    """Plots model mean sqared error for training and test data using matplotlib pockage
    :param history: model fit history
    :return: displays the plot
    """
    print(history.history.keys)
    losses = ['mae', 'val_mae']
    for loss in losses:
        print(loss)
        plt.plot(history.history[loss])
        plt.title('Mean Squared Error Plot')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(losses, loc='best', numpoints=1, fancybox=True)
    plt.show()
    return


def get_test_data(fpath):
    """This function imports test data and reformats the columns in desired format
    :param fpath: path of the training dataset
    :return: training dataframe with Item ID
    """
    df = read_csv(fpath)
    test_df = df.iloc[:, [1, 2, 3, 4, 5, 6, 7]]
    test_ids = df.iloc[:, [0]]
    print(test_df.shape)
    print(test_df.tail(10))
    return test_df, test_ids


def plt_plot_losses(history):
    """Plots model loss for training and test data using matplotlib pockage
    :param history: model fit history
    :return: displays the plot
    """
    losses = ['loss', 'val_loss']
    for loss in losses:
        print(loss)
        plt.plot(history.history[loss])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(losses, loc='best', numpoints=1, fancybox=True)
    plt.show()
    return


def find_missing_cols_after_one_hot_encoding(train_df, test_df, target_col):
    """Finds missing columns between training and test data after performing one hot encoding
    :param train_df: Training dataframe
    :param test_df: Test Dataframe
    :param target_col:Name of Targwt column
    :return:tuple of list of column names
    """
    traincols = (list(train_df.columns))
    testcols = list(test_df.columns)
    cols_not_in_train = []
    for col in testcols:
        if col not in traincols and col != target_col:
            cols_not_in_train.append(col)
    cols_not_in_test = []
    for col in traincols:
        if col not in testcols and col != target_col:
            # print(col)
            cols_not_in_test.append(col)
    return cols_not_in_train, cols_not_in_test


def add_missing_cols(cols_not_in_train, train_df, cols_not_in_test, test_df, target_col):
    """adds missing columns in training and test dataset
    :param cols_not_in_train:
    :param train_df:
    :param cols_not_in_test:
    :param test_df:
    :param target_col:
    :return:
    """
    if len(cols_not_in_train) != 0:
        for col in cols_not_in_train:
            train_df[col] = 0
        sorted_cols = (sorted(train_df.columns))
        train_df = train_df.loc[:, sorted_cols]
        sorted_cols = list(train_df)
        sorted_cols.insert(len(sorted_cols), sorted_cols.pop(sorted_cols.index(target_col)))
        train_df = train_df.ix[:, sorted_cols]

    if len(cols_not_in_test) != 0:
        for col in cols_not_in_test:
            test_df[col] = 0
        sorted_cols = (sorted(test_df.columns))
        test_df = test_df.loc[:, sorted_cols]

    return train_df, test_df


def write_results(preds, val_msle, fpath):
    """Writes the model predictions into a .csv file in a suitable format
    :param preds: np array of model predictions
    :param val_msle: validation mean mean squared logarithmic error
    :param fpath: location of the test file
    :return:
    """

    test_df = read_csv(fpath)
    pred_df = DataFrame(preds)
    pred_df.columns = ['Low_Cap_Price']
    pred_df = concat([test_df, pred_df], axis=1)
    print(list(pred_df.columns))
    pred_df = pred_df[['Item_Id', 'Low_Cap_Price']]
    pred_file = f'Result {datetime.now().strftime("%Y%m%d %H%M")} MSLE {val_msle}.csv'
    print(pred_file)
    pred_df.to_csv(pred_file, index=False)
    return


def calc_accuracy_on_validation_data(preds, fpath):
    val_labels_df = read_csv(fpath)
    val_labels = np.array(val_labels_df)
    msle = round((mean_squared_log_error(val_labels, preds) * 100), 2)
    return msle


def main():
    print('get data')
    base_path = r'C:\Users\akshitagarwal\Desktop\Keras\datasets\prices'
    train_df, train_ids = get_train_data(os.path.join(base_path, 'Train.csv'))
    test_df, test_ids = get_test_data(os.path.join(base_path, 'Test.csv'))
    test_df = prepare_data(test_df)
    check_unique_value(train_df, ['State_of_Country', 'Market_Category', 'Product_Category', 'Grade'])
    train_df = prepare_data(train_df)
    cols_not_in_train, cols_not_in_test = find_missing_cols_after_one_hot_encoding(train_df, test_df,
                                                                                   target_col='Low_Cap_Price')
    train_df, test_df = add_missing_cols(cols_not_in_train, train_df, cols_not_in_test, test_df,
                                         target_col='Low_Cap_Price')
    print(train_df.columns)
    print(test_df.shape)
    x_train, x_test, y_train, y_test = split_dataset(train_df, test_size=0.2, seed=42)
    X_train, Y_train = np.array(x_train), np.array(y_train)

    regressor = get_model(43, 1, magic='relu')
    # regressor = get_baseline_model(43, 1, magic='relu', optimizer='rmsprop', loss=MeanSquaredLogarithmicError(),
    #                                metrics='msle')
    print(regressor.summary())
    history = fit_and_evaluate(regressor, X_train, Y_train, 25, 40, x_test, y_test)
    real_test_data = np.array(test_df)

    plt_plot_losses(history)
    plt_plot_mse(history)

    predictions = regressor.predict(real_test_data)
    msle = round(history.history['val_mae'][-1], 2)
    # msle = calc_accuracy_on_validation_data(predictions, os.path.join(base_path, 'ValidationLabels.csv'))
    write_results(predictions, str(msle), os.path.join(base_path, 'Test.csv'))

    print('done')


if __name__ == '__main__':
    main()
