import os
import argparse
import numpy as np
import tensorflow as tf
from model import GMAN
from utils import load_data, log_string, metric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--P', type=int, default=12, help='history steps')
    parser.add_argument('--Q', type=int, default=12, help='prediction steps')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='testing set ratio')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--traffic_file', default='../data/METR-LA/metr-la.h5', help='traffic file')
    parser.add_argument('--se_file', default='../METR/data/SE(METR).txt', help='spatial embedding file')
    parser.add_argument('--model_file', default='./models/GMAN.weights.h5', help='path to save the model')
    parser.add_argument('--log_file', default='./log/test_log', help='log file')
    parser.add_argument('--time_slot', type=int, default=5, help='time interval')
    parser.add_argument('--K', type=int, default=8, help='number of attention heads')
    parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
    parser.add_argument('--L', type=int, default=5, help='number of STAtt Blocks')
    args = parser.parse_args()

    if not os.path.exists('log'):
        os.makedirs('log')

    log = open(args.log_file, 'w')
    log_string(log, str(args))

    log_string(log, 'loading data...')
    (_, _, _, _, _, _, testX, testTE, testY,
     se, mean, std) = load_data(args)
    log_string(log, f'testX:  {testX.shape}\t\ttestY:  {testY.shape}')
    log_string(log, 'data loaded!')

    log_string(log, 'loading model...')
    model = GMAN(args, se, mean, std)
    model.load_weights(args.model_file)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mape'])
    log_string(log, 'model loaded!')

    log_string(log, 'evaluating...')
    test_metrics = model.evaluate((testX, testTE), testY, batch_size=args.batch_size)
    log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    log_string(log, f'test             {test_metrics[1]:.2f}\t\t{test_metrics[2]:.2f}\t\t{test_metrics[3] * 100:.2f}%')

    log_string(log, 'performance in each prediction step')
    test_pred = model.predict((testX, testTE), batch_size=args.batch_size)
    MAE, RMSE, MAPE = [], [], []
    for q in range(args.Q):
        mae, rmse, mape = metric(test_pred[:, q], testY[:, q])
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        log_string(log, f'step: {q + 1:02d}         {mae:.2f}\t\t{rmse:.2f}\t\t{mape * 100:.2f}%')
    
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    log_string(
        log, f'average:         {average_mae:.2f}\t\t{average_rmse:.2f}\t\t{average_mape * 100:.2f}%'
    )

    log.close()

if __name__ == '__main__':
    main()
