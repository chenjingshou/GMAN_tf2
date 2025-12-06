import tensorflow as tf
print("TensorFlow版本:", tf.__version__)
print("GPU设备列表:", tf.config.list_physical_devices('GPU'))
print("CUDA版本:", tf.sysconfig.get_build_info()["cuda_version"])
print("GPU可用:", tf.test.is_gpu_available())
print("GPU设备列表:", tf.config.list_physical_devices('GPU'))

# 详细GPU信息
if tf.test.is_gpu_available():
    gpu_devices = tf.config.list_physical_devices('GPU')
    for device in gpu_devices:
        print(f"GPU设备: {device}")
        # 测试GPU计算
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("GPU矩阵乘法结果:")
            print(c.numpy())