from pprint import pprint

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
    df = normalize_columns(df, ['Demand', 'Low_Cap_Price', 'High_Cap_Price'], scaler)
    print(df.shape)
    # one hot encoding
    df = one_hot_encode(df, ['State_of_Country', 'Product_Category', 'Grade'])
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


def get_model(input_size, output_size, magic='tanh'):
    """This function creates a baseline feedforward neural network with of given input size and output size
        using magic activation function.
    :param input_size: number of columns in x_train
    :param output_size: no of columns in one hpt
    :param magic: activation function
    :return:Sequential model
    """
    mlmodel = Sequential()
    mlmodel.add(Dense(18, input_dim=input_size, activation=magic))
    # kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l2(1e-4),
    # activity_regularizer=l1(1e-5)))
    # mlmodel.add(LeakyReLU(alpha=0.1))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(256, activation=magic))
    mlmodel.add(Dense(256, activation=magic))
    mlmodel.add(Dense(512, activation=magic))
    mlmodel.add(Dense(512, activation=magic))
    mlmodel.add(Dense(1024, activation=magic))
    mlmodel.add(Dense(output_size, activation='softmax'))

    # Setting optimizer
    # mlmodel.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    opt = SGD(lr=0.001)
    msle = MeanSquaredLogarithmicError()
    mlmodel.compile(loss=msle, optimizer='adam', metrics=['mean_squared_error'])
    return mlmodel


def fit_and_evaluate(model, x_train, y_train, batch_size, epochs):
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

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    print('model fits the input')


def main():
    print('get data')
    test_file = r'C:\Users\akshitagarwal\Desktop\Keras\datasets\prices\Train.csv'
    train_df, train_ids = get_train_data(test_file)
    check_unique_value(train_df, ['State_of_Country', 'Market_Category', 'Product_Category', 'Grade'])
    train_df = prepare_data(train_df)
    print('splitting')
    x_train, x_test, y_train, y_test = split_dataset(train_df, test_size=0.2, seed=42)
    X_train, Y_train = np.array(x_train), np.array(y_train)
    print(X_train.shape)
    print(Y_train.shape)
    regressor = get_model(44, 1, magic='selu')
    print(regressor.summary())
    regressor.fit(X_train, Y_train, batch_size=120, epochs=64, verbose=1)



    
    # past_60_days = orig_training_data.tail(60)
    # df = orig_testing_data.append(past_60_days, ignore_index=True)
    # df = normalize_columns(df, scaler)
    # X_test, Y_test = get_frames(df, 60)
    # X_test, Y_test = np.array(X_test), np.array(Y_test)
    # Y_pred = model.predict(X_test)
    #
    # scale = scaler.scale_[0]
    # Y_pred *= 1 / scale
    # Y_test *= 1 / scale


if __name__ == '__main__':
    main()
