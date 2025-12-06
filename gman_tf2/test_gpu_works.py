# test_gpu_works.py
import tensorflow as tf

print("=" * 70)
print("GPU计算能力测试")
print("=" * 70)

gpus = tf.config.list_physical_devices('GPU')
print(f"检测到 {len(gpus)} 个GPU\n")

for i, gpu in enumerate(gpus):
    print(f"GPU {i}: {gpu}")
    # 设置显存动态增长
    tf.config.experimental.set_memory_growth(gpu, True)

# 在每个GPU上执行测试计算
for i in range(len(gpus)):
    print(f"\n在GPU:{i}上测试矩阵乘法...")
    with tf.device(f'/GPU:{i}'):
        a = tf.random.normal([5000, 5000])
        b = tf.random.normal([5000, 5000])
        c = tf.matmul(a, b)
    print(f"  ✓ GPU:{i} 计算成功！")

print("\n" + "=" * 70)
print("所有GPU测试通过！可以开始训练了！")
print("=" * 70)