import os
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.losses import MeanSquaredLogarithmicError
from keras.optimizers import SGD
from pandas import get_dummies, concat, read_csv, DataFrame, set_option
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_train_data(fpath):
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
    :return: a dictionary
    """
    mydict = {}
    for col in colnames:
        val_count = (df[col].value_counts(dropna=False)).to_dict()
        mydict[col] = val_count
    pprint(mydict)
    return


def prepare_data(df):
    df = df.drop(['Market_Category', 'Date'], axis=1)

    # normalize the columns
    scaler = MinMaxScaler()
    df = normalize_columns(df, ['Demand', 'High_Cap_Price', 'Grade'], scaler)
    print(df.shape)
    # one hot encoding
    # encoder = OneHotEncoder(categories="auto")
    # X_train_encoded = encoder.fit_transform("X_train")
    # X_test_encoded = encoder.transform("X_test")
    df = one_hot_encode(df, ['State_of_Country', 'Product_Category'])
    print(df.shape)
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
    # mlmodel.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    opt = SGD(lr=0.001)
    msle = MeanSquaredLogarithmicError()
    # mlmodel.compile(loss=msle, optimizer='adam', metrics=['mean_squared_error'])
    mlmodel.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])
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
    print(history.history.keys)
    losses = ['mean_squared_error', 'val_mean_squared_error']
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
    df = read_csv(fpath)
    test_df = df.iloc[:, [1, 2, 3, 4, 5, 6, 7]]
    test_ids = df.iloc[:, [0]]
    print(test_df.shape)
    print(test_df.tail(10))
    return test_df, test_ids


def plt_plot_losses(history):
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


def write_results(preds, val_mae, fpath):
    test_df = read_csv(fpath)
    pred_df = DataFrame(preds)
    pred_df.columns = ['Low_Cap_Price']
    pred_df = concat([test_df, pred_df], axis=1)
    print(list(pred_df.columns))
    pred_df = pred_df[['Item_Id', 'Low_Cap_Price']]
    pred_file = f'Result {datetime.now().strftime("%Y%m%d %H%M")} MAE {val_mae}.csv'
    pred_df.to_csv(pred_file, index=False)
    return


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
    x_train, x_test, y_train, y_test = split_dataset(train_df, test_size=0.20, seed=42)
    X_train, Y_train = np.array(x_train), np.array(y_train)
    print(X_train.shape)
    print(Y_train.shape)
    regressor = get_model(44, 1, magic='relu')
    print(regressor.summary())
    history = fit_and_evaluate(regressor, x_train, y_train, 200, 3, x_test, y_test)
    real_test_data = np.array(test_df)

    # plt_plot_losses(history)
    # plt_plot_mse(history)

    predictions = regressor.predict(real_test_data)
    hist = history.history
    print(hist.keys())
    val_mae = int(hist['val_mae'][-1])
    write_results(predictions, str(val_mae), os.path.join(base_path, 'Test.csv'))

    print('done')


if __name__ == '__main__':
    main()
