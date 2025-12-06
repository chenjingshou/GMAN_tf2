import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import GMAN, MaskedMAELoss
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
    parser.add_argument('--SE_file', default='../data/METR-LA/SE(METR).txt', help='spatial embedding file')  # ✅ 改为SE_file
    parser.add_argument('--model_file', default='./models/GMAN.weights.h5', help='path to saved model')
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

    # 检查模型文件是否存在
    if not os.path.exists(args.model_file):
        log_string(log, f'Error: Model file not found: {args.model_file}')
        log.close()
        return

    log_string(log, 'loading data...')
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
     SE, mean, std) = load_data(args)
    
    log_string(log, f'trainX: {trainX.shape}\ttrainY: {trainY.shape}')
    log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:  {testY.shape}')
    log_string(log, 'data loaded!')

    # 创建数据集
    train_ds = tf.data.Dataset.from_tensor_slices(((trainX, trainTE), trainY))
    train_ds = train_ds.batch(args.batch_size)
    
    val_ds = tf.data.Dataset.from_tensor_slices(((valX, valTE), valY))
    val_ds = val_ds.batch(args.batch_size)
    
    test_ds = tf.data.Dataset.from_tensor_slices(((testX, testTE), testY))
    test_ds = test_ds.batch(args.batch_size)

    log_string(log, 'loading model...')
    # ✅ 添加bn参数，与训练时保持一致
    model = GMAN(args, SE, mean, std, bn=True)
    
    # 加载权重前需要先构建模型
    try:
        dummy_batch = next(iter(test_ds.take(1)))
        _ = model(dummy_batch[0], training=False)
        log_string(log, 'model built successfully!')
    except Exception as e:
        log_string(log, f'Warning: Could not build model: {e}')

    # 加载权重
    try:
        model.load_weights(args.model_file)
        log_string(log, f'model weights loaded from {args.model_file}')
    except Exception as e:
        log_string(log, f'Error loading weights: {e}')
        log.close()
        return

    # ✅ 使用正确的loss和optimizer
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=optimizer, 
        loss=MaskedMAELoss(),
        metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse'), 'mape']
    )
    
    log_string(log, 'model compiled!')
    
    # 计算参数量
    try:
        total_params = model.count_params()
        log_string(log, f'trainable parameters: {total_params:,}')
    except:
        pass

    log_string(log, '**** evaluating model ****')
    
    # 评估训练集
    train_metrics = model.evaluate(train_ds, verbose=0)
    log_string(log, f'train - MAE: {train_metrics[1]:.2f}, RMSE: {train_metrics[2]:.2f}, MAPE: {train_metrics[3]*100:.2f}%')
    
    # 评估验证集
    val_metrics = model.evaluate(val_ds, verbose=0)
    log_string(log, f'val   - MAE: {val_metrics[1]:.2f}, RMSE: {val_metrics[2]:.2f}, MAPE: {val_metrics[3]*100:.2f}%')
    
    # 评估测试集
    test_metrics = model.evaluate(test_ds, verbose=0)
    log_string(log, f'test  - MAE: {test_metrics[1]:.2f}, RMSE: {test_metrics[2]:.2f}, MAPE: {test_metrics[3]*100:.2f}%')

    log_string(log, '\n                MAE\t\tRMSE\t\tMAPE')
    log_string(log, f'train            {train_metrics[1]:.2f}\t\t{train_metrics[2]:.2f}\t\t{train_metrics[3]*100:.2f}%')
    log_string(log, f'val              {val_metrics[1]:.2f}\t\t{val_metrics[2]:.2f}\t\t{val_metrics[3]*100:.2f}%')
    log_string(log, f'test             {test_metrics[1]:.2f}\t\t{test_metrics[2]:.2f}\t\t{test_metrics[3]*100:.2f}%')

    log_string(log, '\nperformance in each prediction step')
    
    # 获取测试集预测结果
    test_pred = model.predict(test_ds, verbose=0)
    
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
    print(f'\nTest completed! Results saved to {args.log_file}')

if __name__ == '__main__':
    main()