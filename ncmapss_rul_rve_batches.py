import utils
import mdcl_utils
from argparse import ArgumentParser
import atexit
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Lambda, LSTM, RepeatVector, Bidirectional, Masking, Dropout
import os
import gc
import h5py
import time
import model
import matplotlib
import matplotlib.ticker as ticker
import numpy as np
from os.path import join as opj, exists as ope, realpath as opr, splitext as ops, basename as opb
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
    logger = logging.getLogger("root.condition_scaler")
    scaler = StandardScaler()
    for condition in df['op_cond'].unique():
        logger.info(list(df['op_cond'].unique()).index(condition))
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
    # test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit'] == unit_nr], sequence_length, sensors, -99.))
    #             for unit_nr in X_test_pre['unit'].unique())
    # x_test = np.concatenate(list(test_gen)).astype(np.float32)
    x_test = gen_data_wrapper(X_test_pre, sequence_length, sensors, X_test_pre.unit.unique())
    y_test = gen_label_wrapper(df_y_test, sequence_length, ['RUL'], X_test_pre.unit.unique())

    return x_train, y_train, x_val, y_val, x_test, y_test

def evaluate(y_true, y_hat, label='test'):
    logger = logging.getLogger("root.evaluate")
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    logger.info('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))


def score(y_true, y_hat):
    res = 0
    for true, hat in zip(y_true, y_hat):
        subs = hat - true
        if subs < 0:
            res = res + np.exp(-subs / 13)[0] - 1
        else:
            res = res + np.exp(subs / 10)[0] - 1
    return res


def arg_parser():
    """
    This function is used to process inputs given to the program
    """
    DESCRIPTION = "This script uses RVAE-Rgeressor to predict RUL of a NASA Turbofan engine"
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-gen_np_data', help="Generate and save data at ./RULRVE_numpydata_scriptgen",
                        action='store_true')
    # parser.add_argument('-outDir', help='Run directory', required=True, type=path_arg)
    parser.add_argument('-technology', help="Technology")
    parser.add_argument('-lis', help="log file", default=ops(opb(__file__))[0]+".lis")
    parser.add_argument('-DEBUG', action='store_true')
    parser.add_argument('-retain_gds_ascii', action='store_true')

    return parser.parse_args()

def gen_numpy_data(dataset_dir, out_dir):
    logger = logging.getLogger("root.gen_numpy_data")
    logger.info("Created directory to store numpy data @ {}".format(opr('./RULRVE_numpydata_scriptgen')))

    file_list = ['N-CMAPSS_DS01-005.h5', 'N-CMAPSS_DS02-006.h5', 'N-CMAPSS_DS03-012.h5',
                 'N-CMAPSS_DS04.h5', 'N-CMAPSS_DS05.h5', 'N-CMAPSS_DS06.h5', 'N-CMAPSS_DS07.h5',
                 'N-CMAPSS_DS08a-009.h5', 'N-CMAPSS_DS08c-008.h5']
    for i, filename in enumerate(file_list):
        logger.info("Loading data from: {}".format(filename))
        # filename = "N-CMAPSS_DS01-005.h5"
        df_W_Xs_dev, df_y_dev, df_W_Xs_test, df_y_test = load_data(opj(dataset_dir, filename))
        sensors = ['T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc',
                   'Wf', ]
        logger.info("Preparing data for: {}".format(filename))
        x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(
            sequence_length, sensors, df_W_Xs_dev, df_y_dev, df_W_Xs_test, df_y_test)
        logger.info("Saving data in numpy format")
        filename_suffix = ops(filename)[0].split('_')[-1]
        np.save('RULRVE_numpydata_scriptgen/x_train_{}.npy'.format(filename_suffix), x_train)
        del (x_train)
        np.save('RULRVE_numpydata_scriptgen/x_val_{}.npy'.format(filename_suffix), x_val)
        del (x_val)
        np.save('RULRVE_numpydata_scriptgen/y_train_{}.npy'.format(filename_suffix), y_train)
        del (y_train)
        np.save('RULRVE_numpydata_scriptgen/y_val_{}.npy'.format(filename_suffix), y_val)
        del (y_val)
        np.save('RULRVE_numpydata_scriptgen/x_test_{}.npy'.format(filename_suffix), x_test)
        del (x_test)
        np.save('RULRVE_numpydata_scriptgen/y_test_{}.npy'.format(filename_suffix), y_test)
        del (y_test)

    return


def main():
    # filename = "/data/adas_vision_data1/users/adithya/turbofan_ncmapss/dataset/N-CMAPSS_DS01-005.h5"
    args = arg_parser()
    gen_np_data = args.gen_np_data
    lis = args.lis
    logger = mdcl_utils.command_display(lis, args.DEBUG)

    if gen_np_data:
        if not ope('./RULRVE_numpydata_scriptgen'):
            os.mkdir('./RULRVE_numpydata_scriptgen')
        dataset_dir = "/data/adas_vision_data1/users/adithya/turbofan_ncmapss/dataset/"
        out_dir = './RULRVE_numpydata_scriptgen'
        gen_numpy_data(dataset_dir, out_dir)
        logger.info("Finished generating numpy data. Exiting")
        sys.exit(0)

    else:
        def numpy_loader(filepath):
            logger = logging.getLogger("root.numpy_loader")
            with open(filepath, 'rb') as fp:
                np_ndarray = np.load(fp)
            logger.info("{} has a numpy ndarray of shape: {}".format(filepath, np_ndarray.shape))
            return np_ndarray



        # x_train = np.concatenate((x_train1, x_train2))
        # x_val = np.concatenate((x_val1, x_val2))
        # y_train = np.concatenate((y_train1, y_train2))
        # y_val = np.concatenate((y_val1, y_val2))
        # x_test = np.concatenate((x_test1, x_test2))
        # y_test = pd.concat((y_test1, y_test2))

    sequence_length = 30
    # smoothing intensity
    alpha = 0.1
    # max RUL

    # ----------------------------- MODEL -----------------------------------
    threshold = 100
    # no_of_samples = x_train.shape[0]
    timesteps = 30 # x_train1.shape[1]
    input_dim = 15 # x_train2.shape[2]
    intermediate_dim = 300
    batch_size = 512 # 1024
    latent_dim = 2
    epochs = 1000 # 1000
    optimizer = 'adam'
    strategy = tf.distribute.MirroredStrategy()
    # atexit.register(strategy._extended._collective_ops._pool.close)  # type: ignore
    with strategy.scope():
        RVE = model.create_model(timesteps, input_dim, intermediate_dim,
                                 batch_size, latent_dim, epochs, optimizer, )

    # -----------------------------------------------------------------------

    # --------------------------- TRAINING ---------------------------------
    # Callbacks for training
    data_dir = '/data/adas_vision_data1/users/adithya/turbofan_ncmapss/RULRVE_numpydata_scriptgen'
    x_val, y_val = None, None
    for i in [1, 2, 3, 4, 5]: # [1, 2, 3, 4, 5]:
        x_train = numpy_loader(opj(data_dir, 'x_train{}.npy'.format(i)))
        x_val = numpy_loader(opj(data_dir, 'x_val{}.npy'.format(i)))
        y_train = numpy_loader(opj(data_dir, 'y_train{}.npy'.format(i)))
        y_val = numpy_loader(opj(data_dir, 'y_val{}.npy'.format(i)))

        model_callbacks = utils.get_callbacks(RVE, x_train, y_train)
        results = RVE.fit(x_train, y_train,
                          shuffle=True,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(x_val, y_val),
                          callbacks=model_callbacks, verbose=2)
        del(x_train)
        del(x_val)
        del(y_train)
        del(y_val)
        gc.collect()
    logger.info("Finished fitting")

    # RVE.save_weights('ncmapss_123_rve_fit_model')
    #logger.info("Saved model")
    # RVE.load_weights('ncmapss_123_rve_fit_model')
    #RVE.load_weights(tf.train.latest_checkpoint("./checkpoints/"))

    train_mu = np.array([], dtype=np.float32).reshape(0, 2)
    test_mu = np.array([], dtype=np.float32).reshape(0, 2)

    for i in [1, 2, 3, 4, 5]: # [1, 2, 3, 4, 5]:
        x_train = numpy_loader(opj(data_dir, 'x_train{}.npy'.format(i)))
        x_val = numpy_loader(opj(data_dir, 'x_val{}.npy'.format(i)))
        y_train = numpy_loader(opj(data_dir, 'y_train{}.npy'.format(i)))
        y_val = numpy_loader(opj(data_dir, 'y_val{}.npy'.format(i)))
        # for x_sub_array, y_sub_array in zip(
        #         np.array_split(x_train, 1000),
        #         np.array_split(y_train, 1000)):
        for x_sub_array, y_sub_array in zip(
                np.array_split(np.concatenate((x_train, x_val)), 1000),
                np.array_split(np.concatenate((y_train, y_val)), 1000)):
            z, _, _ = RVE.encoder(x_sub_array, y_sub_array)
            # logger.info(z.shape)
            train_mu = np.concatenate((train_mu, z))

        del (x_train)
        del (x_val)
        del (y_train)
        del (y_val)
        gc.collect()

        x_test = numpy_loader(opj(data_dir, 'x_test{}_total.npy'.format(i)))
        y_test = numpy_loader(opj(data_dir, 'y_test{}_total.npy'.format(i)))
        # y_test = y_test.reshape(len(y_test), 1)
        for x_sub_array, y_sub_array in zip(np.array_split(x_test, 1000), np.array_split(y_test, 1000)):
            z, _, _ = RVE.encoder(x_sub_array, y_sub_array)
            test_mu = np.concatenate((test_mu, z))
        del(x_test)
        del(y_test)
        gc.collect()
    y_hat_train = RVE.regressor.predict(train_mu)
    y_hat_test = RVE.regressor.predict(test_mu)
    # nasa_score = 0
    y_dev_total = np.array([], dtype=np.float32).reshape(0, 1)
    y_test_total = np.array([], dtype=np.float32).reshape(0, 1)
    for i in [1, 2, 3, 4, 5]: # [1, 2, 3, 4, 5]:
        y_train = numpy_loader(opj(data_dir, 'y_train{}.npy'.format(i)))
        y_dev_total = np.concatenate((y_dev_total, y_train))
        del(y_train)
        y_val = numpy_loader(opj(data_dir, 'y_val{}.npy'.format(i)))
        y_dev_total = np.concatenate((y_dev_total, y_val))
        del(y_val)
        y_test = numpy_loader(opj(data_dir, 'y_test{}_total.npy'.format(i)))
        # y_test = y_test.reshape(len(y_test), 1)
        y_test_total = np.concatenate((y_test_total, y_test))
        del(y_test)
        gc.collect()
    utils.evaluate(y_dev_total, y_hat_train, 'train')
    del(y_dev_total)
    del(y_hat_train)
    gc.collect()
    # utils.evaluate(y_train, y_hat_train, 'train')
    utils.evaluate(y_test_total, y_hat_test, 'test')
    utils.score(y_test_total, y_hat_test)
        # del (y_train)
        # del (y_val)
        # del (y_test)
        # del (y_test)
    # logger.info("NASA score: ", nasa_score)
    atexit.register(strategy._extended._collective_ops._pool.close)


if __name__ == "__main__":
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    main()
