import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import GMAN, MaskedMAELoss
from utils import load_data, log_string, metric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_slot', type=int, default=5, help='time interval')
    parser.add_argument('--P', type=int, default=12, help='history steps')
    parser.add_argument('--Q', type=int, default=12, help='prediction steps')
    parser.add_argument('--L', type=int, default=5, help='number of STAtt Blocks')
    parser.add_argument('--K', type=int, default=8, help='number of attention heads')
    parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='testing set ratio')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--traffic_file', default='../data/METR-LA/metr-la.h5', help='traffic file')
    parser.add_argument('--SE_file', default='../data/METR-LA/SE(METR).txt', help='spatial embedding file')
    parser.add_argument('--model_file', default='./models/GMAN.weights.h5', help='saved model path')
    parser.add_argument('--log_file', default='./log/test_log', help='log file')
    args = parser.parse_args()

    if not os.path.exists('log'):
        os.makedirs('log')

    # 检查模型文件是否存在
    if not os.path.exists(args.model_file):
        print(f'Error: Model file not found: {args.model_file}')
        return

    with open(args.log_file, 'w') as log:
        log_string(log, str(args))
        log_string(log, f'Model file: {args.model_file}')
        
        # 加载数据
        log_string(log, 'loading data...')
        (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
         SE, mean, std) = load_data(args)
        
        log_string(log, f'trainX: {trainX.shape}\ttrainY: {trainY.shape}')
        log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
        log_string(log, f'testX:  {testX.shape}\t\ttestY:  {testY.shape}')
        log_string(log, 'data loaded!')

        # 创建数据集
        train_ds = tf.data.Dataset.from_tensor_slices(((trainX, trainTE), trainY))
        train_ds = train_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices(((valX, valTE), valY))
        val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices(((testX, testTE), testY))
        test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

        # 构建模型
        log_string(log, 'building model...')
        model = GMAN(args, SE, mean, std, bn=True)
        
        # 构建模型（通过虚拟前向传播）
        dummy_batch = next(iter(test_ds.take(1)))
        _ = model(dummy_batch[0], training=False)
        log_string(log, 'model built!')

        # 加载权重
        log_string(log, f'loading weights from {args.model_file}...')
        model.load_weights(args.model_file)
        log_string(log, 'weights loaded!')

        # 测试
        log_string(log, '\n**** evaluating model ****')
        
        # 训练集评估
        log_string(log, 'evaluating on training set...')
        train_pred = model.predict(train_ds, verbose=0)
        train_mae, train_rmse, train_mape = metric(train_pred, trainY)
        
        # 验证集评估
        log_string(log, 'evaluating on validation set...')
        val_pred = model.predict(val_ds, verbose=0)
        val_mae, val_rmse, val_mape = metric(val_pred, valY)
        
        # 测试集评估
        log_string(log, 'evaluating on test set...')
        test_pred = model.predict(test_ds, verbose=0)
        test_mae, test_rmse, test_mape = metric(test_pred, testY)
        
        # 输出总体结果
        log_string(log, '\n                MAE\t\tRMSE\t\tMAPE')
        log_string(log, f'train            {train_mae:.2f}\t\t{train_rmse:.2f}\t\t{train_mape * 100:.2f}%')
        log_string(log, f'val              {val_mae:.2f}\t\t{val_rmse:.2f}\t\t{val_mape * 100:.2f}%')
        log_string(log, f'test             {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%')

        # 每个时间步的详细结果
        log_string(log, '\nperformance in each prediction step')
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
        log_string(log, f'average:         {average_mae:.2f}\t\t{average_rmse:.2f}\t\t{average_mape * 100:.2f}%')
        
        log_string(log, '\nTest completed!')
        print(f'\nResults saved to {args.log_file}')

if __name__ == '__main__':
    main()