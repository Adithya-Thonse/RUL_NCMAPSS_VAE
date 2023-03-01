import utils
import mdcl_utils
from argparse import ArgumentParser
import atexit
import logging
import tensorflow as tf
import os
import gc
import new_model as model
import numpy as np
from os.path import join as opj, splitext as ops, basename as opb
from sklearn.metrics import r2_score, mean_squared_error
total_file_list = [
    'N-CMAPSS_DS01-005.h5', 'N-CMAPSS_DS02-006.h5', 'N-CMAPSS_DS03-012.h5',
    'N-CMAPSS_DS04.h5', 'N-CMAPSS_DS05.h5', 'N-CMAPSS_DS06.h5', 'N-CMAPSS_DS07.h5',
    'N-CMAPSS_DS08a-009.h5', 'N-CMAPSS_DS08c-008.h5']


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
    parser.add_argument('-out_dir', help='Run directory', type=mdcl_utils.path_arg,
                        default=os.getcwd())
    parser.add_argument('-splits', help="Split each dataset into n subarrays", default=10, type=int)
    parser.add_argument('-saved_numpy_data_dir', help='Processed numpy directory', type=mdcl_utils.path_arg,
                        default="/home/a0484689/PycharmProjects/Remaining-Useful-Life-Estimation-Variational/split_scaled_numpy_arrays_10/")
    parser.add_argument('-batch_size', help="Batch Size", default=1024, type=int)
    parser.add_argument('-epochs', help="Max Epochs. (Early stopping is present)", default=1000, type=int)
    parser.add_argument('-dataset_numbers', nargs='+', type=int,
                        help=("Combination of datasets to load. Enter space separated numbers between 0-8"
                              "0-8 indicate indices of: {}".format(', '.join(total_file_list))))
    parser.add_argument('-load_weights_from', help='Instead of training, use checkpoint directory to load weights',
                        type=mdcl_utils.path_arg)

    parser.add_argument('-lis', help="log file", default=ops(opb(__file__))[0]+".lis")
    parser.add_argument('-DEBUG', action='store_true')
    parser.add_argument('-just_train', action='store_true')
    return parser.parse_args()


def numpy_loader(filepath):
    logger = logging.getLogger("root.numpy_loader")
    with open(filepath, 'rb') as fp:
        np_ndarray = np.load(fp)
    logger.info("{} has a numpy ndarray of shape: {}".format(filepath, np_ndarray.shape))
    return np_ndarray


def main():
    # filename = "/data/adas_vision_data1/users/adithya/turbofan_ncmapss/dataset/N-CMAPSS_DS01-005.h5"
    args = arg_parser()
    lis = args.lis
    logger = mdcl_utils.command_display(lis, args.DEBUG)
    if args.dataset_numbers:
        file_list = [total_file_list[idx] for idx in args.dataset_numbers]
    else:
        file_list = total_file_list
    filename_suffix_list = [ops(filename)[0].split('_')[-1] for filename in file_list]

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
    batch_size = args.batch_size # 1024
    latent_dim = 2
    epochs = args.epochs # 1000
    optimizer = 'adam'
    strategy = tf.distribute.MirroredStrategy()
    # atexit.register(strategy._extended._collective_ops._pool.close)  # type: ignore
    with strategy.scope():
        RVE = model.create_model(timesteps, input_dim, intermediate_dim,
                                 batch_size, latent_dim, epochs, optimizer, )
        # RVE.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    # -----------------------------------------------------------------------

    # --------------------------- TRAINING ---------------------------------
    # Callbacks for training
    data_dir = args.saved_numpy_data_dir
    x_val, y_val = None, None

    if args.load_weights_from:
        # default = opj(os.getcwd(), "checkpoints")
        logger.info("Loading weights from: {}".format(args.load_weights_from))
        RVE.load_weights(tf.train.latest_checkpoint(args.load_weights_from))
    else:

        for filename_suffix in filename_suffix_list:
            x_val = numpy_loader(opj(data_dir, 'x_val_{}.npy'.format(filename_suffix)))
            y_val = numpy_loader(opj(data_dir, 'y_val_{}.npy'.format(filename_suffix)))
            for subarray_num in range(args.splits):
                sub_array_num = str(subarray_num).zfill(3)
                x_train = numpy_loader(opj(data_dir, 'x_train_{}_{}.npy'.format(filename_suffix, sub_array_num)))
                # x_val = numpy_loader(opj(data_dir, 'x_val_{}_{}.npy'.format(filename_suffix, sub_array_num)))
                y_train = numpy_loader(opj(data_dir, 'y_train_{}_{}.npy'.format(filename_suffix, sub_array_num)))
                # y_val = numpy_loader(opj(data_dir, 'y_val_{}_{}.npy'.format(filename_suffix, sub_array_num)))

                model_callbacks = utils.get_callbacks(RVE, x_train, y_train)
                results = RVE.fit(x_train, y_train,
                                  shuffle=True,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  validation_data=(x_val, y_val),
                                  callbacks=model_callbacks, verbose=2)
                del x_train
                del y_train
                gc.collect()
            del y_val
            del x_val
            gc.collect()
        logger.info("Finished fitting")


    train_mu = np.array([], dtype=np.float32).reshape(0, 2)
    val_mu = np.array([], dtype=np.float32).reshape(0, 2)
    test_mu = np.array([], dtype=np.float32).reshape(0, 2)
    '''
    for filename_suffix in filename_suffix_list:
        for subarray_num in range(args.splits):
            sub_array_num = str(subarray_num).zfill(3)
            x_train = numpy_loader(opj(data_dir, 'x_train_{}_{}.npy'.format(filename_suffix, sub_array_num)))
            # x_val = numpy_loader(opj(data_dir, 'x_val_{}_{}.npy'.format(filename_suffix, sub_array_num)))
            y_train = numpy_loader(opj(data_dir, 'y_train_{}_{}.npy'.format(filename_suffix, sub_array_num)))
            # y_val = numpy_loader(opj(data_dir, 'y_val_{}_{}.npy'.format(filename_suffix, sub_array_num)))

            # for x_sub_array, y_sub_array in zip(
            #         np.array_split(x_train, 1000),
            #         np.array_split(y_train, 1000)):
            for x_sub_array, y_sub_array in zip(
                    np.array_split(np.concatenate((x_train, x_val)), 1000),
                    np.array_split(np.concatenate((y_train, y_val)), 1000)):
                z, _, _ = RVE.encoder(x_sub_array, y_sub_array)
                # logger.info(z.shape)
                train_mu = np.concatenate((train_mu, z))

            del x_train
            del y_train
            gc.collect()
    for filename_suffix in filename_suffix_list:
        x_val = numpy_loader(opj(data_dir, 'x_val_{}.npy'.format(filename_suffix)))
        y_val = numpy_loader(opj(data_dir, 'y_val_{}.npy'.format(filename_suffix)))
        del x_val
        del y_val
        gc.collect()
    '''
    test_mu = np.array([], dtype=np.float32).reshape(0, 1)
    y_test_total = np.array([], dtype=np.float32).reshape(0, 1)
    for filename_suffix in filename_suffix_list:
        for subarray_num in range(args.splits):
            sub_array_num = str(subarray_num).zfill(3)
            x_test = numpy_loader(opj(data_dir, 'x_test_{}_{}.npy'.format(filename_suffix, sub_array_num)))[::100, ...]
            y_test = numpy_loader(opj(data_dir, 'y_test_{}_{}.npy'.format(filename_suffix, sub_array_num)))[::100, ...]
            # y_test = y_test.reshape(len(y_test), 1)
            # for x_sub_array, y_sub_array in zip(np.array_split(x_test, 1000), np.array_split(y_test, 1000)):
            #     z, _, _ = RVE.encoder(x_sub_array, y_sub_array)
            #     test_mu = np.concatenate((test_mu, z))

            y_pred = RVE(x_test)
            test_mu = np.concatenate((test_mu, y_pred))
            y_test_total = np.concatenate((y_test_total, y_test))
            del x_test
            del y_test
            gc.collect()
    if args.just_train:
        atexit.register(strategy._extended._pcollective_ops._pool.close)
        sys.exit(0)


    # y_hat_train = RVE.regressor.predict(train_mu)
    # y_hat_test = RVE.regressor.predict(test_mu)
    # nasa_score = 0
    # y_dev_total = np.array([], dtype=np.float32).reshape(0, 1)
    # y_test_total = np.array([], dtype=np.float32).reshape(0, 1)
    # for filename_suffix in filename_suffix_list:
    #     # y_train = numpy_loader(opj(data_dir, 'y_train_{}.npy'.format(filename_suffix)))
    #     # y_dev_total = np.concatenate((y_dev_total, y_train))
    #     # del y_train
    #     # y_val = numpy_loader(opj(data_dir, 'y_val_{}.npy'.format(filename_suffix)))
    #     # y_dev_total = np.concatenate((y_dev_total, y_val))
    #     # del y_val
    #     y_test = numpy_loader(opj(data_dir, 'y_test_{}.npy'.format(filename_suffix)))
    #     # y_test = y_test.reshape(len(y_test), 1)
    #     y_test_total = np.concatenate((y_test_total, y_test))
    #     del y_test
    #     gc.collect()
    # utils.evaluate(y_dev_total, y_hat_train, 'train')
    # del y_dev_total
    # del y_hat_train
    gc.collect()
    # utils.evaluate(y_test_total, y_hat_test, 'test')
    # utils.score(y_test_total, y_hat_test)

    utils.evaluate(y_test_total, test_mu, 'test')
    utils.score(y_test_total, test_mu)
    print(len(test_mu))
    print(len(y_test_total))
    # print(len(y_test_total))

    # logger.info("NASA score: ", nasa_score)

    atexit.register(strategy._extended._pcollective_ops._pool.close)


if __name__ == "__main__":
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    main()
