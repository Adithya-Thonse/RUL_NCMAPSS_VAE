import utils
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Lambda, LSTM, RepeatVector, Bidirectional, Masking, Dropout
import os
import h5py
import time
import model
import matplotlib
import matplotlib.ticker as ticker
import numpy as np
from os.path import join as opj
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflowkeras.layers import Dense
# from keras.layers import Dropout
from sklearn.model_selection import GroupShuffleSplit

def add_operating_condition(df):
    df_op_cond = df.copy()
    # 'Fc', 'alt', 'Mach',  'TRA'
    df_op_cond['TRA'] = abs(df_op_cond['TRA'].round()).astype(int)
#     df_op_cond['Mach'] = abs(df_op_cond['Mach'].round(decimals=2))

    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['Fc'].astype(str) + '_' + \
                            df_op_cond['Mach'].round(decimals=1).astype(str) + '_' + \
                            df_op_cond['TRA'].astype(str)
#     (df_op_cond['alt']//1000).astype(str) + '_' + \
    df_op_cond['op_cond'] = df_op_cond.op_cond.astype(str)

    return df_op_cond

def condition_scaler(df, sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    for condition in df['op_cond'].unique():
        print(list(df['op_cond'].unique()).index(condition))
#         scaler.fit(df.loc[df['op_cond'] == condition, sensor_names])
        df.loc[df['op_cond'] == condition, sensor_names] = scaler.fit_transform(df.loc[df['op_cond'] == condition, sensor_names])
    return df

def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    # first, take the exponential weighted mean
    df[sensors] = df.groupby('unit')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)

    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    mask = df.groupby('unit')['unit'].transform(create_mask, samples=n_samples).astype(bool)
    df = df[mask]

    return df

def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 and +1 because of Python indexing
    for start, stop in zip(range(0, num_elements - (sequence_length - 1)), range(sequence_length, num_elements + 1)):
        yield data[start:stop, :]


def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit'].unique()

    data_gen = (list(gen_train_data(df[df['unit'] == unit_nr], sequence_length, columns))
                for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array


def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length - 1:num_elements, :]


def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit'].unique()

    label_gen = [gen_labels(df[df['unit'] == unit_nr], sequence_length, label)
                 for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array

def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value)  # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:, :] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values

    # specifically yield the last possible sequence
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]

def load_data(filename):
    df_X_cols = ['alt', 'Mach', 'TRA', 'T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2',
                 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf', 'unit', 'cycle']

    df_y_cols = ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod',
                 'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod',
                 'LPT_eff_mod', 'LPT_flow_mod', 'unit', 'alt', 'cycle']

    X = []
    # y = []

    df_X = pd.DataFrame(X, columns=df_X_cols)

    # Load data
    with h5py.File(filename, 'r') as hdf:
        # Development set
        W_dev = np.array(hdf.get('W_dev'))  # W
        X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
        #         X_v_dev = np.array(hdf.get('X_v_dev'))         # X_v
        #         T_dev = np.array(hdf.get('T_dev'))             # T
        Y_dev = np.array(hdf.get('Y_dev'))  # RUL
        A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))  # W
        X_s_test = np.array(hdf.get('X_s_test'))  # X_s
        #         X_v_test = np.array(hdf.get('X_v_test'))       # X_v
        #         T_test = np.array(hdf.get('T_test'))           # T
        Y_test = np.array(hdf.get('Y_test'))  # RUL
        A_test = np.array(hdf.get('A_test'))  # Auxiliary

        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        #         X_v_var = np.array(hdf.get('X_v_var'))
        #         T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))
        #         X_v_var = list(np.array(X_v_var, dtype='U20'))
        #         T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))
        W_Xs_var = W_var + X_s_var
    W_Xs_dev = np.concatenate((W_dev, X_s_dev), axis=1)
    W_Xs_test = np.concatenate((W_test, X_s_test), axis=1)

    df_A_dev = DataFrame(data=A_dev, columns=A_var)
    df_A_test = DataFrame(data=A_test, columns=A_var)

    df_W_Xs_dev = DataFrame(data=W_Xs_dev, columns=W_Xs_var)
    df_W_Xs_test = DataFrame(data=W_Xs_test, columns=W_Xs_var)
    #     df_T = DataFrame(data=T, columns=T_var)

    # Add the column "unit" to all dataframes
    #     df_T['unit'] = df_A['unit'].values
    #     df_T['alt'] = df_W_Xs['alt'].values
    df_W_Xs_dev['unit'] = df_A_dev['unit'].values
    df_W_Xs_test['unit'] = df_A_test['unit'].values

    #     df_T['cycle'] = df_A['cycle'].values
    df_W_Xs_dev['cycle'] = df_A_dev['cycle'].values
    df_W_Xs_test['cycle'] = df_A_test['cycle'].values

    df_W_Xs_dev['Fc'] = df_A_dev['Fc'].values
    df_W_Xs_test['Fc'] = df_A_test['Fc'].values
    #     df_W_Xs['HS'] = df_A['hs'].values

    cols = ['unit', 'cycle', 'Fc', 'alt', 'Mach', 'TRA', 'T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24',
            'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf', ]
    df_W_Xs_dev = df_W_Xs_dev[cols]
    df_W_Xs_test = df_W_Xs_test[cols]

    for col in df_W_Xs_dev:
        if col in ['unit', 'cycle', 'alt', 'Fc']:
            df_W_Xs_dev[col] = df_W_Xs_dev[col].astype(int)
            df_W_Xs_test[col] = df_W_Xs_test[col].astype(int)
        else:
            df_W_Xs_dev[col] = df_W_Xs_dev[col].astype(float)
            df_W_Xs_test[col] = df_W_Xs_test[col].astype(float)

    df_y_dev = pd.DataFrame(zip(df_W_Xs_dev.unit, Y_dev.reshape(len(Y_dev))), columns=['unit', 'RUL'])
    df_y_test = pd.DataFrame(zip(df_W_Xs_test.unit, Y_test.reshape(len(Y_test))), columns=['unit', 'RUL'])

    return df_W_Xs_dev, df_y_dev, df_W_Xs_test, df_y_test

def prepare_data(sequence_length, sensors, df_W_Xs_dev, df_y_dev, df_W_Xs_test, df_y_test):
    # remove unused sensors
    sensor_names = ['T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc',
                    'Wf', ]
    drop_sensors = [element for element in sensor_names if element not in sensors]

    X_train_pre = add_operating_condition(df_W_Xs_dev.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(df_W_Xs_test.drop(drop_sensors, axis=1))

    X_train_pre = condition_scaler(X_train_pre, sensors)
    X_test_pre = condition_scaler(X_test_pre, sensors)

    X_train_pre = exponential_smoothing(X_train_pre, sensors, 0, alpha=0.1)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha=0.1)

    # train-val split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.87, random_state=42)
    # generate the train/val for *each* sample -> for that we iterate over the train and val units we want
    # this is a for that iterates only once and in that iterations at the same time iterates over all the values we want,
    # i.e. train_unit and val_unit are not a single value but a set of training/vali units
    for train_unit, val_unit in gss.split(X_train_pre['unit'].unique(), groups=X_train_pre['unit'].unique()):
        train_unit = X_train_pre['unit'].unique()[train_unit]  # gss returns indexes and index starts at 1
        val_unit = X_train_pre['unit'].unique()[val_unit]

        x_train = gen_data_wrapper(X_train_pre, sequence_length, sensors, train_unit)
        y_train = gen_label_wrapper(df_y_dev, sequence_length, ['RUL'], train_unit)

        x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
        y_val = gen_label_wrapper(df_y_dev, sequence_length, ['RUL'], val_unit)

    # create sequences for test
    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit'] == unit_nr], sequence_length, sensors, -99.))
                for unit_nr in X_test_pre['unit'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, df_y_test.RUL


def main():
    filename = "/data/adas_vision_data1/users/adithya/turbofan_ncmapss/dataset/N-CMAPSS_DS01-005.h5"
    load_saved_data = True
    if load_saved_data:
        def loader(filepath):
            with open(filepath, 'rb') as fp:
                np_ndarray = np.load(fp)
            print("{} has a numpy ndarray of shape: {}".format(filepath, np_ndarray.shape))
            return np_ndarray

        data_dir = '/data/adas_vision_data1/users/adithya/turbofan_ncmapss/RULRVE_numpydata_total'
        # data_dir = '/data/adas_vision_data1/users/adithya/turbofan_ncmapss/RULRVE_numpydata_1_val_eng'
        x_train1 = loader(opj(data_dir, 'x_train1.npy'))
        x_val1 = loader(opj(data_dir, 'x_val1.npy'))
        y_train1 = loader(opj(data_dir, 'y_train1.npy'))
        y_val1 = loader(opj(data_dir, 'y_val1.npy'))
        # x_val = x_val[:700000, ...]
        # y_val = y_val[:700000, ...]
        x_test1 = loader(opj(data_dir, 'x_test1.npy'))
        df_y_test1 = pd.read_csv(opj(data_dir, 'df_y_test1.csv'))
        y_test1 = df_y_test1.RUL


        x_train2 = loader(opj(data_dir, 'x_train2.npy'))
        x_val2 = loader(opj(data_dir, 'x_val2.npy'))
        y_train2 = loader(opj(data_dir, 'y_train2.npy'))
        y_val2 = loader(opj(data_dir, 'y_val2.npy'))
        # x_val = x_val[:700000, ...]
        # y_val = y_val[:700000, ...]
        x_test2 = loader(opj(data_dir, 'x_test2.npy'))
        df_y_test2 = pd.read_csv(opj(data_dir, 'df_y_test2.csv'))
        y_test2 = df_y_test2.RUL

        # x_train = np.concatenate((x_train1, x_train2))
        # x_val = np.concatenate((x_val1, x_val2))
        # y_train = np.concatenate((y_train1, y_train2))
        # y_val = np.concatenate((y_val1, y_val2))
        # x_test = np.concatenate((x_test1, x_test2))
        # y_test = pd.concat((y_test1, y_test2))


    else:
        sequence_length = 30
        # smoothing intensity
        alpha = 0.1
        # max RUL
        threshold = 100
        df_W_Xs_dev, df_y_dev, df_W_Xs_test, df_y_test = load_data(filename)
        sensors = ['T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf', ]
        x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(
            sequence_length, sensors, df_W_Xs_dev, df_y_dev, df_W_Xs_test, df_y_test)

    # ----------------------------- MODEL -----------------------------------
    # no_of_samples = x_train.shape[0]
    timesteps = x_train1.shape[1]
    input_dim = x_train2.shape[2]
    intermediate_dim = 300
    batch_size = 2048 # 1024
    latent_dim = 2
    epochs = 30 # 1000
    optimizer = 'adam'
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        RVE = model.create_model(timesteps, input_dim, intermediate_dim,
                                 batch_size, latent_dim, epochs, optimizer, )
    # Callbacks for training
    model_callbacks1 = utils.get_callbacks(RVE, x_train1, y_train1)
    model_callbacks2 = utils.get_callbacks(RVE, x_train2, y_train2)


    # -----------------------------------------------------------------------

    # --------------------------- TRAINING ---------------------------------
    results1 = RVE.fit(x_train1, y_train1,
                      shuffle=True,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(x_val1, y_val1),
                      callbacks=model_callbacks1, verbose=2)
    results2 = RVE.fit(x_train2, y_train2,
                      shuffle=True,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(x_val2, y_val2),
                      callbacks=model_callbacks2, verbose=2)

    # train_mu = RVE.encoder.predict(np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)))
    # test_mu = RVE.encoder.predict(x_test, y_test.clip(upper=threshold))
    # train_mu = RVE.encoder.predict(np.concatenate((x_train, x_val)), batch_size=8)
    # print(train_mu.shape)
    # test_mu = RVE.encoder.predict(x_test, batch_size=8)

    # train_mu = []
    # test_mu = []
    # for i, array in enumerate(np.array_split(x_test, 100)):
    #     print("{}/100".format(i))
    #     z, _, _ = RVE.encoder.predict(array)
    #     test_mu.extend(z)
    # for i, array in enumerate(np.array_split(np.concatenate((x_train, x_val)), 100)):
    #     print("{}/100".format(i))
    #     z, _, _ = RVE.encoder.predict(array)
    #     train_mu.extend(z)
    #
    # # print(test_mu.shape)
    # # Evaluate
    # y_hat_train = RVE.regressor.predict(train_mu)
    # y_hat_test = RVE.regressor.predict(test_mu)
    #
    # utils.evaluate(np.concatenate((y_train, y_val)), y_hat_train, 'train')
    # utils.evaluate(y_test, y_hat_test, 'test')

if __name__ == "__main__":
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    main()
