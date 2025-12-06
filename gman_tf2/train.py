import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras  # 关键修复：从tf导入keras

# 某些 TensorFlow 版本在布局优化（NHWC<->NCHW转换）阶段会触发 "Size of values 0 does not match size of permutation 4"。
# 关闭 layout optimizer 可绕过该问题，避免训练中途报错。
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
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
    parser.add_argument('--max_epoch', type=int, default=100, help='epoch to run')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=5, help='decay epoch')
    parser.add_argument('--traffic_file', default='../data/METR-LA/metr-la.h5', help='traffic file')
    parser.add_argument('--SE_file', default='../data/METR-LA/SE(METR).txt', help='spatial embedding file')
    parser.add_argument('--model_file', default='./models/GMAN.weights.h5', help='save the model to disk')
    parser.add_argument('--log_file', default='./log/log', help='log file')
    parser.add_argument('--use_bn', type=bool, default=True, help='use batch normalization')
    args = parser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('log'):
        os.makedirs('log')

    with open(args.log_file, 'w') as log:
        log_string(log, str(args))
        
        # 检查GPU可用性
        gpus = tf.config.list_physical_devices('GPU')
        log_string(log, f'检测到 {len(gpus)} 个GPU设备')
        if gpus:
            for gpu in gpus:
                log_string(log, f'  - {gpu}')
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    log_string(log, f'设置内存增长失败: {e}')
        else:
            log_string(log, '警告：未检测到GPU，将使用CPU训练（速度较慢）')
        
        log_string(log, 'loading data...')
        (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
         SE, mean, std) = load_data(args)
        log_string(log, f'trainX: {trainX.shape}\ttrainY: {trainY.shape}')
        log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
        log_string(log, f'testX:  {testX.shape}\t\ttestY:  {testY.shape}')
        log_string(log, f'SE:     {SE.shape}')
        log_string(log, f'mean: {mean:.2f}, std: {std:.2f}')
        log_string(log, 'data loaded!')

        # tf.data pipeline
        train_ds = tf.data.Dataset.from_tensor_slices(((trainX, trainTE), trainY))
        train_ds = train_ds.cache().shuffle(buffer_size=2048).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(((valX, valTE), valY))
        val_ds = val_ds.cache().batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = tf.data.Dataset.from_tensor_slices(((testX, testTE), testY))
        test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

        # 混合精度训练：仅在有GPU时启用
        if gpus:
            # log_string(log, '启用混合精度训练（mixed_float16）')
            # policy = keras.mixed_precision.Policy('mixed_float16')
            # keras.mixed_precision.set_global_policy(policy)
            log_string(log, '使用默认精度（float32）')
        else:
            log_string(log, '使用默认精度（float32）')

        log_string(log, 'compiling model...')
        model = GMAN(args, SE, mean, std, bn=True)

        # 学习率衰减调度器
        num_train = trainX.shape[0]
        steps_per_epoch = num_train // args.batch_size
        decay_steps = args.decay_epoch * steps_per_epoch

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=decay_steps,
            decay_rate=0.7,
            staircase=True
        )

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)

        if gpus and keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
            log_string(log, '使用LossScaleOptimizer包装优化器')

        model.compile(
            optimizer=optimizer, 
            loss=MaskedMAELoss(),
            metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse'), 'mape']
        )

        # 关键修复：通过虚拟前向传播构建模型
        log_string(log, 'building model...')
        try:
            dummy_batch = next(iter(train_ds.take(1)))
            _ = model(dummy_batch[0], training=False)
            log_string(log, 'model built successfully!')
        except Exception as e:
            log_string(log, f'Warning: Could not build model: {e}')
            log_string(log, 'Model will be built during first training step')

        # 计算参数量（如果模型已构建）
        try:
            total_params = model.count_params()
            log_string(log, f'trainable parameters: {total_params:,}')
        except ValueError:
            log_string(log, 'Model not built yet, parameter count will be available after first epoch')

        log_string(log, '**** training model ****')
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True
        )
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=args.model_file,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        
        # 学习率记录回调
        class LRLogger(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                lr = self.model.optimizer.learning_rate
                if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
                    lr = lr(self.model.optimizer.iterations)
                log_string(log, f'epoch {epoch+1}, learning rate: {float(lr):.6f}')
        
        history = model.fit(
            train_ds,
            epochs=args.max_epoch,
            validation_data=val_ds,
            callbacks=[early_stopping_callback, model_checkpoint_callback, LRLogger()],
            verbose=1
        )

        log_string(log, '**** testing model ****')
        model.load_weights(args.model_file)
        log_string(log, 'model restored!')
        log_string(log, 'evaluating...')

        test_pred = model.predict(test_ds)
        test_mae, test_rmse, test_mape = metric(test_pred.reshape(-1, args.Q, testY.shape[-1]), testY)
        
        log_string(log, '                MAE\t\tRMSE\t\tMAPE')
        log_string(log, f'test             {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%')

        log_string(log, 'performance in each prediction step')
        test_pred = model.predict(test_ds)
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


if __name__ == '__main__':
    main()