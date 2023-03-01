from argparse import ArgumentParser
import gc
import h5py
import logging
import mdcl_utils
import numpy as np
import os
from os.path import join as opj, exists as ope, realpath as opr, splitext as ops, basename as opb
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import sys

total_file_list = [
    'N-CMAPSS_DS01-005.h5', 'N-CMAPSS_DS02-006.h5', 'N-CMAPSS_DS03-012.h5',
    'N-CMAPSS_DS04.h5', 'N-CMAPSS_DS05.h5', 'N-CMAPSS_DS06.h5', 'N-CMAPSS_DS07.h5',
    'N-CMAPSS_DS08a-009.h5', 'N-CMAPSS_DS08c-008.h5']

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
    unique_op_cond = df['op_cond'].unique()
    for condition in unique_op_cond:
        logger.debug("Condition Number: {}/{}".format(
            list(unique_op_cond).index(condition), len(list(unique_op_cond))-1))
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
    # this for iterates only once and in that iterations at the same time iterates over all the values we want,
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


def arg_parser():
    """
    This function is used to process inputs given to the program
    """
    DESCRIPTION = "This script preprocesses and generates intermediate data of a NASA Turbofan engine from the dataset"
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-out_dir', help='Run directory', type=mdcl_utils.path_arg,
                        default=os.getcwd())
    parser.add_argument('-dataset_dir', help='h5 dataset directory', type=mdcl_utils.path_arg,
                        default="/data/adas_vision_data1/users/adithya/turbofan_ncmapss/")
    parser.add_argument('-sequence_length', help="Sequence Length", default=30, type=int)
    parser.add_argument('-dataset_numbers', nargs='+', type=int,
                        help=("Combination of datasets to load. Enter space separated numbers between 0-8"
                              "0-8 indicate indices of: {}".format(', '.join(total_file_list))))
    parser.add_argument('-lis', help="log file", default=ops(opb(__file__))[0]+".lis")
    parser.add_argument('-DEBUG', action='store_true')
    return parser.parse_args()


def gen_numpy_data(dataset_dir, out_dir, file_list, sequence_length=30):
    logger = logging.getLogger("root.gen_numpy_data")
    logger.info("Created directory to store numpy data @ {}".format(out_dir))

    for i, filename in enumerate(file_list):
        logger.info("Loading data from: {}".format(filename))
        # filename = "N-CMAPSS_DS01-005.h5"
        df_W_Xs_dev, df_y_dev, df_W_Xs_test, df_y_test = load_data(opj(dataset_dir, filename))
        sensors = ['T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf',]
        logger.info("Preparing data for: {}".format(filename))
        x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(
            sequence_length, sensors, df_W_Xs_dev, df_y_dev, df_W_Xs_test, df_y_test)
        logger.info("Saving data in numpy format")
        filename_suffix = ops(filename)[0].split('_')[-1]
        np.save(opj(out_dir, 'x_train_{}.npy'.format(filename_suffix)), x_train)
        del x_train
        np.save(opj(out_dir, 'x_val_{}.npy'.format(filename_suffix)), x_val)
        del x_val
        np.save(opj(out_dir, 'y_train_{}.npy'.format(filename_suffix)), y_train)
        del y_train
        np.save(opj(out_dir, 'y_val_{}.npy'.format(filename_suffix)), y_val)
        del y_val
        np.save(opj(out_dir, 'x_test_{}.npy'.format(filename_suffix)), x_test)
        del x_test
        np.save(opj(out_dir, 'y_test_{}.npy'.format(filename_suffix)), y_test)
        del y_test

        gc.collect()
    return


def main():
    args = arg_parser()
    lis = args.lis
    logger = mdcl_utils.command_display(lis, args.DEBUG)
    out_dir = opj(args.out_dir, 'RULRVE_numpydata_scriptgen')
    if not ope(out_dir):
        os.mkdir(out_dir)
    dataset_dir = args.dataset_dir

    if args.dataset_numbers:
        file_list = [total_file_list[idx] for idx in args.dataset_numbers]
    else:
        file_list = total_file_list
    logger.info("Datasets selected: {}".format(', '.join(file_list)))
    gen_numpy_data(dataset_dir, out_dir, file_list, args.sequence_length)
    logger.info("Finished generating numpy data at {} ".format(out_dir))
    sys.exit(0)


if __name__ == "__main__":
    main()
