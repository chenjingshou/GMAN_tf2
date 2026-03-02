import os
import argparse
import time
import numpy as np
import tensorflow as tf
import keras
from model import MaskedClippedMAPE, GMAN, MaskedMAELoss
from utils import load_data, log_string, metric
from keras import mixed_precision
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 保持混合精度设置与训练时一致
mixed_precision.set_global_policy("mixed_float16")

def describe_model_and_weights(model, model_file):
    """
    模型验证环节：检查文件信息，并对比加载前后的权重变化
    """
    print("\n🕵️‍♀️ [Model Verification] Inspecting...")
    
    # 1. 检查文件元数据
    if not os.path.exists(model_file):
        print(f"❌ Error: Weight file not found at {model_file}")
        return False
    
    stats = os.stat(model_file)
    mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_mtime))
    file_size_mb = stats.st_size / (1024 * 1024)
    print(f"   📄 File: {model_file}")
    print(f"   🕒 Last Modified: {mod_time}")
    print(f"   📦 Size: {file_size_mb:.2f} MB")

    # 2. 选取一个特定的层进行采样（Spatial Embedding层通常不为0）
    target_layer = None
    for layer in model.layers:
        if 'st_embedding' in layer.name:
            target_layer = layer
            break
    
    if target_layer:
        # 获取加载前的权重快照
        weights_before = target_layer.get_weights()
        if len(weights_before) > 0:
            sample_before = weights_before[0].flatten()[:5] # 取前5个值
            print(f"   📉 Weights (Random Init): {sample_before}")
        
        print(f">> Loading weights from {model_file}...")
        try:
            model.load_weights(model_file)
        except ValueError as e:
            print(f"❌ Load Failed! Parameter mismatch? Error: {e}")
            return False

        # 获取加载后的权重
        weights_after = target_layer.get_weights()
        if len(weights_after) > 0:
            sample_after = weights_after[0].flatten()[:5]
            print(f"   📈 Weights (Trained):     {sample_after}")
            
            # 简单验证：权重是否发生变化
            if np.array_equal(sample_before, sample_after):
                print("⚠️ Warning: Weights did NOT change. Did loading work?")
            else:
                print("✅ Verification PASSED: Weights successfully updated from file!")
    else:
        print("   ⚠️ Could not find 'st_embedding' layer for verification, strictly loading only.")
        model.load_weights(model_file)
        print("✅ Weights loaded.")

    return True

def main():
    parser = argparse.ArgumentParser()
    # ⚠️ 这里的默认值我改回了你训练时用的 L=8, K=4, d=4，以防报错
    parser.add_argument('--time_slot', type=int, default=5)
    parser.add_argument('--P', type=int, default=12)
    parser.add_argument('--Q', type=int, default=12)
    parser.add_argument('--L', type=int, default=8, help='CHANGED to 8 to match trained weights')
    parser.add_argument('--K', type=int, default=4, help='CHANGED to 4 to match trained weights')
    parser.add_argument('--d', type=int, default=4, help='CHANGED to 4 to match trained weights')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    # 测试相关参数
    parser.add_argument('--refresh_stats_ratio', type=float, default=0.05)
    parser.add_argument('--infer_ensemble', type=int, default=1)
    parser.add_argument("--use_ln", default=True, type=bool)
    parser.add_argument('--traffic_file', default='data/METR.h5')
    parser.add_argument('--SE_file', default='data/SE(METR).txt')
    parser.add_argument('--model_file', default='./models/GMAN.weights.h5')
    
    # 即使测试不用，为了构建Graph也要占位
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--block_dropout', type=float, default=0.0)
    
    args = parser.parse_args()

    # 1. 加载数据
    print(">> Loading Data...")
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std) = load_data(args)

    # 2. 初始化模型
    print(f">> Initializing GMAN (L={args.L}, K={args.K}, d={args.d})...")
    model = GMAN(args, SE, mean, std, bn=True)

    # 3. Build Graph (必须先跑一次数据才能load weights)
    print(">> Building Model Graph (Dummy Inference)...")
    def make_test_ds(x):
        return tf.data.Dataset.from_tensor_slices(((x, testTE), testY)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    dummy_input = next(iter(make_test_ds(testX)))[0]
    _ = model(dummy_input, training=False)

    # 4. 描述并加载权重 (你的需求)
    if not describe_model_and_weights(model, args.model_file):
        return

    # 5. [逻辑复刻] 刷新统计数据 Refresh Norm Stats
    if args.refresh_stats_ratio > 0:
        print(f"\n>> Logic Check: Refreshing Normalization Stats (ratio={args.refresh_stats_ratio})...")
        refresh_n = max(1, int(testX.shape[0] * args.refresh_stats_ratio))
        # 还原回 Raw Data
        ref_raw = (testX[:refresh_n] * std) + mean
        new_mean = float(np.mean(ref_raw))
        new_std = float(np.std(ref_raw))
        if new_std < 1e-6: new_std = 1e-6
        
        print(f"   Old Mean/Std: {mean:.4f} / {std:.4f}")
        print(f"   New Mean/Std: {new_mean:.4f} / {new_std:.4f} (Computed from first {refresh_n} samples)")

        # 使用新参数处理测试集
        test_raw = (testX * std) + mean
        testX = (test_raw - new_mean) / new_std
        test_ds = make_test_ds(testX)

        # 更新模型内的 Mean/Std
        model.mean = tf.constant(new_mean, dtype=tf.float32)
        model.std = tf.constant(new_std, dtype=tf.float32)
    else:
        test_ds = make_test_ds(testX)

    # 6. 推理
    print("\n>> Starting Inference...")
    start_time = time.time()
    test_pred = model.predict(test_ds, verbose=1)
    
    if args.infer_ensemble > 1:
        preds = [test_pred]
        for i in range(args.infer_ensemble - 1):
            print(f"   Ensemble run {i+2}/{args.infer_ensemble}...")
            preds.append(model.predict(test_ds, verbose=0))
        test_pred = np.mean(preds, axis=0)
    
    end_time = time.time()
    print(f">> Inference Time: {end_time - start_time:.2f}s")

    # 7. 评估
    print(">> Calculating Metrics...")
    # reshape to (Sample, Q, Node)
    mae, rmse, mape = metric(test_pred.reshape(-1, args.Q, testY.shape[-1]), testY)
    
    print(f'\n🏆 ================================== 🏆')
    print(f'   Final Test Results (Loaded Weights)')
    print(f'   MAE : {mae:.4f}')
    print(f'   RMSE: {rmse:.4f}')
    print(f'   MAPE: {mape*100:.4f}%')
    print(f'🏆 ================================== 🏆')

if __name__ == '__main__':
    main()