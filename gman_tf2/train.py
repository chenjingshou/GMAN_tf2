import os
import argparse
import numpy as np
import tensorflow as tf
import keras
from model import MaskedClippedMAPE
from model import GMAN, MaskedMAELoss
from utils import load_data, log_string, metric
from keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def main():
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--time_slot', type=int, default=5, help='time interval')
    parser.add_argument('--P', type=int, default=12, help='history steps')
    parser.add_argument('--Q', type=int, default=12, help='prediction steps')
    parser.add_argument('--L', type=int, default=5, help='number of STAtt Blocks')
    parser.add_argument('--K', type=int, default=8, help='number of attention heads')
    parser.add_argument('--d', type=int, default=8, help='dims of each head')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='testing set ratio')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=1000, help='epoch to run')
    parser.add_argument('--patience', type=int, default=16, help='patience for early stop')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=5, help='decay epoch')
    # 数据增强 - 时空联合遮挡与噪声
    parser.add_argument('--aug_enable', type=bool, default=True, help='enable spatiotemporal augmentation on training set')
    parser.add_argument('--aug_block_prob', type=float, default=0.2, help='probability to apply time-node block masking')
    parser.add_argument('--aug_block_t', type=int, default=2, help='time length of masked block')
    parser.add_argument('--aug_block_n', type=int, default=8, help='node count of masked block')
    parser.add_argument('--aug_noise_std', type=float, default=0.01, help='stddev of additive Gaussian noise')
    parser.add_argument('--aug_drift_std', type=float, default=0.01, help='stddev of global additive drift')
    parser.add_argument('--aug_trend_std', type=float, default=0.01, help='stddev of linear time trend')
    parser.add_argument('--aug_rw_std', type=float, default=0.02, help='stddev of random-walk drift per step')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout for MultiHeadAttention')
    parser.add_argument('--block_dropout', type=float, default=0.1, help='stochastic depth drop rate per STAtt block')
    parser.add_argument('--infer_ensemble', type=int, default=1, help='number of forward passes to ensemble at inference (>=1)')
    parser.add_argument('--refresh_stats_ratio', type=float, default=0.05, help='fraction of test set used to recompute norm stats for inference (0 to disable)')
    parser.add_argument("--use_ln", default=True, type=bool, help="use Layer Normalization instead of Batch Normalization")
    # [Task 6] 服务器路径配置
    parser.add_argument('--traffic_file', default='data/METR.h5', help='traffic file')
    parser.add_argument('--SE_file', default='data/SE(METR).txt', help='spatial embedding file')
    
    parser.add_argument('--model_file', default='./models/GMAN.weights.h5', help='save the model to disk')
    parser.add_argument('--log_file', default='./log/log', help='log file')
    args = parser.parse_args()

    if not os.path.exists('models'): os.makedirs('models')
    if not os.path.exists('log'): os.makedirs('log')
    os.makedirs('logs/fit', exist_ok=True)

    with open(args.log_file, 'w') as log:
        log_string(log, str(args))
        
        # GPU 配置
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try: tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e: log_string(log, str(e))
        
        log_string(log, 'loading data...')
        # 加载数据 (trainX被全局归一化，trainY保持Raw值)
        (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std) = load_data(args)
        log_string(log, f'trainX: {trainX.shape}\ttrainY: {trainY.shape}')
        log_string(log, f'SE shape: {SE.shape}')
        
        # 数据集构建与增强
        def augment_fn(inputs, y):
            X, TE = inputs  # X: (P, N)
            X = tf.cast(X, tf.float32)
            if args.aug_enable:
                # 时间-节点联合遮挡
                def mask_block(x):
                    P = tf.shape(x)[0]; N = tf.shape(x)[1]
                    t_len = tf.minimum(args.aug_block_t, P)
                    n_len = tf.minimum(args.aug_block_n, N)
                    max_t = tf.maximum(P - t_len, 1)
                    max_n = tf.maximum(N - n_len, 1)
                    t0 = tf.random.uniform([], 0, max_t, dtype=tf.int32)
                    n0 = tf.random.uniform([], 0, max_n, dtype=tf.int32)
                    tm = tf.concat([tf.ones([t0], x.dtype), tf.zeros([t_len], x.dtype), tf.ones([P - t0 - t_len], x.dtype)], axis=0)
                    nm = tf.concat([tf.ones([n0], x.dtype), tf.zeros([n_len], x.dtype), tf.ones([N - n0 - n_len], x.dtype)], axis=0)
                    mask = tm[:, None] * nm[None, :]
                    return x * mask
                X = tf.cond(tf.random.uniform([], 0, 1) < args.aug_block_prob, lambda: mask_block(X), lambda: X)
                # 轻量噪声
                if args.aug_noise_std > 0:
                    noise = tf.random.normal(tf.shape(X), stddev=tf.cast(args.aug_noise_std, X.dtype))
                    X = X + noise
                # 全局漂移（空间一致的偏移）
                if args.aug_drift_std > 0:
                    drift = tf.random.normal([], stddev=tf.cast(args.aug_drift_std, X.dtype))
                    X = X + drift
                # 时间线性趋势（模拟慢速漂移）
                if args.aug_trend_std > 0:
                    P = tf.shape(X)[0]
                    t_lin = tf.linspace(tf.cast(0., X.dtype), tf.cast(1., X.dtype), P)
                    slope = tf.random.normal([], stddev=tf.cast(args.aug_trend_std, X.dtype))
                    X = X + slope * t_lin[:, None]
                # 随机游走漂移（方向和速率随机变化）
                if args.aug_rw_std > 0:
                    P = tf.shape(X)[0]
                    steps = tf.random.normal([P], stddev=tf.cast(args.aug_rw_std, X.dtype))
                    rw = tf.cumsum(steps)
                    X = X + rw[:, None]
            return (X, TE), y

        train_ds = tf.data.Dataset.from_tensor_slices(((trainX, trainTE), trainY))
        train_ds = train_ds.shuffle(2048)
        if args.aug_enable:
            train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(((valX, valTE), valY))
        val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        def make_test_ds(x):
            return tf.data.Dataset.from_tensor_slices(((x, testTE), testY)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = make_test_ds(testX)

        log_string(log, 'compiling model...')
        # 初始化模型 (Task 6 Enhanced)
        model = GMAN(args, SE, mean, std, bn=True)
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=args.decay_epoch * (trainX.shape[0] // args.batch_size),
            decay_rate=0.7, staircase=True
        )
        # 针对 Raw Data 输出，必须限制梯度，防止震荡
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=5.0)
        

        model.compile(
            optimizer=optimizer, 
            loss=MaskedMAELoss(),
            metrics=['mae', MaskedClippedMAPE(), keras.metrics.RootMeanSquaredError(name='rmse')],
            # jit_compile=args.jit_compile
        )

        # 触发 build 以便显示参数量 (RevIN 权重在 __init__ 中已创建，这里主要为了 GMAN 其他部分)
        try:
            _ = model(next(iter(train_ds.take(1)))[0], training=False)
            model.summary(print_fn=lambda x: log_string(log, x))
        except Exception as e:
            log_string(log, f"Model build skipped: {e}")

        log_string(log, '**** training model ****')
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(args.model_file, save_best_only=True, save_weights_only=True),
            keras.callbacks.TensorBoard(log_dir='logs/fit', update_freq='epoch')
        ]
        
        model.fit(train_ds, epochs=args.max_epoch, validation_data=val_ds, callbacks=callbacks, verbose=1)

        log_string(log, '**** testing model ****')
        model.load_weights(args.model_file)

        if args.refresh_stats_ratio > 0:
            refresh_n = max(1, int(testX.shape[0] * args.refresh_stats_ratio))
            ref_raw = (testX[:refresh_n] * std) + mean
            new_mean = float(np.mean(ref_raw))
            new_std = float(np.std(ref_raw))
            if new_std < 1e-6:
                log_string(log, f'Refreshed std too small ({new_std:.6f}); clamped to 1e-6')
                new_std = 1e-6
            log_string(log, f'Refresh norm stats from first {refresh_n} / {testX.shape[0]} test samples: mean={new_mean:.4f}, std={new_std:.4f}')

            test_raw = (testX * std) + mean
            testX = (test_raw - new_mean) / new_std
            test_ds = make_test_ds(testX)

            model.mean = tf.constant(new_mean, dtype=tf.float32)
            model.std = tf.constant(new_std, dtype=tf.float32)
        else:
            log_string(log, 'Test-time norm stats refresh disabled; using training stats for inference.')
        
        test_pred = model.predict(test_ds, verbose=0)
        if args.infer_ensemble > 1:
            preds = [test_pred]
            for _ in range(args.infer_ensemble - 1):
                preds.append(model.predict(test_ds, verbose=0))
            test_pred = np.mean(preds, axis=0)
        # 这里的 test_pred 已经是 Raw Values，直接评估
        mae, rmse, mape = metric(test_pred.reshape(-1, args.Q, testY.shape[-1]), testY)
        log_string(log, f'Final Results: MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape*100:.2f}%')

if __name__ == '__main__':
    main()
