import utils
import mdcl_utils
from argparse import ArgumentParser
import atexit
import logging
import tensorflow as tf
import os
import gc
import model
import numpy as np
from os.path import join as opj, splitext as ops, basename as opb
from sklearn.metrics import r2_score, mean_squared_error


total_file_list = [
    'N-CMAPSS_DS01-005.h5', 'N-CMAPSS_DS02-006.h5', 'N-CMAPSS_DS03-012.h5',
    'N-CMAPSS_DS04.h5', 'N-CMAPSS_DS05.h5', 'N-CMAPSS_DS06.h5', 'N-CMAPSS_DS07.h5',
    'N-CMAPSS_DS08a-009.h5', 'N-CMAPSS_DS08c-008.h5']


def arg_parser():
    """
    This function is used to process inputs given to the program
    """
    DESCRIPTION = "This script uses RVAE-Rgeressor to predict RUL of a NASA Turbofan engine"
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-out_dir', help='Run directory', type=mdcl_utils.path_arg,
                        default=os.getcwd())
    parser.add_argument('-saved_numpy_data_dir', help='Processed numpy directory', type=mdcl_utils.path_arg,
                        default="/data/adas_vision_data1/users/adithya/turbofan_ncmapss/RULRVE_numpydata_scriptgen")
    parser.add_argument('-splits', help="Split each dataset into n subarrays", default=10, type=int)
    parser.add_argument('-dataset_numbers', nargs='+', type=int,
                        help=("Combination of datasets to load. Enter space separated numbers between 0-8"
                              "0-8 indicate indices of: {}".format(', '.join(total_file_list))))

    parser.add_argument('-lis', help="log file", default=ops(opb(__file__))[0]+".lis")
    parser.add_argument('-DEBUG', action='store_true')
    return parser.parse_args()


def numpy_loader(filepath):
    logger = logging.getLogger("root.numpy_loader")
    with open(filepath, 'rb') as fp:
        np_ndarray = np.load(fp)
    logger.info("{} has a numpy ndarray of shape: {}".format(filepath, np_ndarray.shape))
    return np_ndarray

def array_splitter(nd_array, out_dir, filename, splits):
    logger = logging.getLogger("root.array_splitter")
    for i, sub_array in enumerate(np.array_split(nd_array, splits)):
        filepath = opj(out_dir, "{}_{}.npy".format(ops(opb(filename))[0], str(i).zfill(3)))
        np.save(filepath, sub_array)
        logger.info("{} has a numpy ndarray of shape: {}".format(filepath, sub_array.shape))


def main():
    # filename = "/data/adas_vision_data1/users/adithya/turbofan_ncmapss/dataset/N-CMAPSS_DS01-005.h5"
    args = arg_parser()
    lis = args.lis
    logger = mdcl_utils.command_display(lis, args.DEBUG)
    logger = mdcl_utils.command_display(lis, args.DEBUG)
    data_dir = args.saved_numpy_data_dir
    if args.dataset_numbers:
        file_list = [total_file_list[idx] for idx in args.dataset_numbers]
    else:
        file_list = total_file_list
    filename_suffix_list = [ops(filename)[0].split('_')[-1] for filename in file_list]

    for filename_suffix in filename_suffix_list:
        for array_type in ['x_train', 'x_val', 'x_test', 'y_train', 'y_val', 'y_test']:
            filename = opj(data_dir, '{}_{}.npy'.format(array_type, filename_suffix))
            loaded_array = numpy_loader(filename)
            array_splitter(loaded_array, args.out_dir, filename, args.splits)

        # x_val = numpy_loader(opj(data_dir, 'x_val_{}.npy'.format(filename_suffix)))
        # y_train = numpy_loader(opj(data_dir, 'y_train_{}.npy'.format(filename_suffix)))
        # y_val = numpy_loader(opj(data_dir, 'y_val_{}.npy'.format(filename_suffix)))

if __name__ == "__main__":
    main()