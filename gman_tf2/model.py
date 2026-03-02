import tensorflow as tf
from keras.layers import Layer, Dropout, BatchNormalization, LayerNormalization, Dense, MultiHeadAttention
import keras

# =============================================================================
#  [Task 6 核心模块] RevIN: 解决时空分布偏移 (已修复混合精度兼容性)
# =============================================================================
class RevIN(Layer):
    def __init__(self, num_nodes, num_features=1, eps=1e-5, affine=True, **kwargs):
        """
        :param num_nodes: 节点数量 (用于解决空间异质性)
        """
        super(RevIN, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        # [关键修复] 在 __init__ 中直接创建权重，避免 Lazy Loading Bug
        if self.affine:
            # Shape: (1, 1, Nodes, Features)
            # 为每个节点学习独立的缩放(Scale)和偏置(Bias)，解决空间分布不一致问题
            self.affine_weight = self.add_weight(name='affine_weight', 
                                                 shape=(1, 1, self.num_nodes, self.num_features),
                                                 initializer='ones', trainable=True)
            self.affine_bias = self.add_weight(name='affine_bias', 
                                               shape=(1, 1, self.num_nodes, self.num_features),
                                               initializer='zeros', trainable=True)

    def normalize(self, x):
        # x shape: (Batch, Time, Nodes, Features)
        # x.dtype 在混合精度下通常是 float16
        
        # [Fix MP] 1. 创建掩码：忽略 0 值 (缺失值)
        # 使用 cast(0.0, x.dtype) 确保比较类型一致
        mask = tf.not_equal(x, tf.cast(0.0, x.dtype))
        
        # [Fix MP] 关键修改：将 mask 转换为与 x 相同的类型 (float16 或 float32)
        mask_cast = tf.cast(mask, x.dtype)
        
        # [Fix MP] 2. 计算有效值的数量
        valid_count = tf.reduce_sum(mask_cast, axis=1, keepdims=True)
        # 确保分母也是 x.dtype
        valid_count = tf.maximum(valid_count, tf.cast(1.0, x.dtype)) 
        
        # [Fix MP] 3. 计算 Masked Mean
        sum_x = tf.reduce_sum(x * mask_cast, axis=1, keepdims=True)
        mean = sum_x / valid_count
        
        # [Fix MP] 4. 计算 Masked Variance -> Std
        diff = (x - mean) * mask_cast
        variance = tf.reduce_sum(tf.square(diff), axis=1, keepdims=True) / valid_count
        stdev = tf.sqrt(variance + tf.cast(self.eps, x.dtype))
        
        # [Fix MP] 5. 执行归一化 (保持 0 值位置仍为 0)
        x_norm = (x - mean) / stdev
        x_norm = tf.where(mask, x_norm, tf.zeros_like(x_norm))
        
        # 仿射变换
        if self.affine:
            # 确保参数也被 cast 到正确的类型 (通常 TF 会自动广播，但显式 cast 更安全)
            w = tf.cast(self.affine_weight, x.dtype)
            b = tf.cast(self.affine_bias, x.dtype)
            x_norm = x_norm * w + b
            
        return x_norm, mean, stdev

    def denormalize(self, x, mean, stdev):
        # 1. 反向仿射变换
        if self.affine:
            w = tf.cast(self.affine_weight, x.dtype)
            b = tf.cast(self.affine_bias, x.dtype)
            # 加上极小值防止除零，同样要做类型转换
            eps = tf.cast(1e-10, x.dtype)
            x = (x - b) / (w + eps)
        
        # 2. 恢复原始的时间分布统计量
        x_denorm = x * stdev + mean
        return x_denorm
    
    def call(self, x):
        x_norm, _, _ = self.normalize(x)
        return x_norm

# =============================================================================
#  基础组件（使用 Keras 优化实现）
# =============================================================================
class DenseBlock(Layer):
    """堆叠 Dense + 可选 Norm/Dropout，输入保持最后一维特征不改变 rank。"""
    def __init__(self, units, activations, drop=None, bn=False, ln=False):
        super().__init__()
        if isinstance(units, int): units = [units]; activations = [activations]
        elif isinstance(units, tuple): units = list(units); activations = list(activations)
        self.bn = bn; self.ln = ln; self.drop = drop
        self.denses = []
        self.bn_layers = []
        for num_unit, activation in zip(units, activations):
            self.denses.append(Dense(num_unit, activation=None, kernel_initializer='glorot_uniform'))
            if activation is not None and (bn or ln):
                self.bn_layers.append(BatchNormalization(momentum=0.9) if bn else LayerNormalization())
            else:
                self.bn_layers.append(None)
        if self.drop is not None: self.dropout_layer = Dropout(drop)
        self.activations = activations

    def call(self, x, training=None):
        for dense, bn_layer, activation in zip(self.denses, self.bn_layers, self.activations):
            if self.drop is not None: x = self.dropout_layer(x, training=training)
            x = dense(x)
            if activation is not None:
                if bn_layer is not None: x = bn_layer(x, training=training)
                x = activation(x)
        return x

class MaskedMAELoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        # 确保 mask 和计算类型一致
        compute_dtype = y_pred.dtype
        y_true = tf.cast(y_true, compute_dtype)
        
        mask = tf.not_equal(y_true, tf.cast(0.0, compute_dtype))
        mask = tf.cast(mask, compute_dtype)
        
        mask_mean = tf.maximum(tf.reduce_mean(mask), tf.cast(1e-6, compute_dtype))
        mask = mask / mask_mean
        
        mae = tf.abs(y_pred - y_true) * mask
        return tf.cast(tf.reduce_mean(mae), tf.float32)


class MaskedClippedMAPE(keras.metrics.Metric):
    """MAPE with zero-mask and denominator clipping to avoid explosion."""
    def __init__(self, eps=1.0, name="masked_mape", **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.not_equal(y_true, 0.0)
        denom = tf.maximum(tf.abs(y_true), self.eps)
        mape = tf.abs((y_pred - y_true) / denom)
        mape = tf.where(mask, mape, 0.0)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            mape *= sample_weight
            mask = tf.cast(mask, tf.float32) * sample_weight
        self.total.assign_add(tf.reduce_sum(mape))
        self.count.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class STEmbedding(Layer):
    def __init__(self, D, bn=False, ln=False):
        super(STEmbedding, self).__init__()
        self.fc_se = DenseBlock(units=[D, D], activations=[tf.nn.relu, None], bn=bn, ln=ln)
        self.fc_te = DenseBlock(units=[D, D], activations=[tf.nn.relu, None], bn=bn, ln=ln)
    def call(self, SE, TE, T, training=None):
        SE = tf.expand_dims(tf.expand_dims(SE, axis=0), axis=0)
        SE = self.fc_se(SE, training=training)
        dayofweek = tf.one_hot(TE[..., 0], depth=7)
        timeofday = tf.one_hot(TE[..., 1], depth=T)
        TE = tf.concat((dayofweek, timeofday), axis=-1)
        TE = self.fc_te(tf.expand_dims(TE, axis=2), training=training)
        return tf.add(SE, TE)

class SpatialAttention(Layer):
    """使用内置 MultiHeadAttention，注意力轴为节点维度。"""
    def __init__(self, K, d, bn=False, ln=False, attn_drop=0.0):
        super(SpatialAttention, self).__init__()
        self.K = K; self.d = d; self.D = K * d
        self.mha = MultiHeadAttention(num_heads=K, key_dim=d, attention_axes=(1,), output_shape=self.D, dropout=attn_drop)
        self.fc = DenseBlock([self.D, self.D], [tf.nn.relu, None], bn=bn, ln=ln)
    def call(self, X, STE, training=None):
        # X shape: (B, T, N, D)
        H = tf.concat((X, STE), axis=-1)
        B, T, N, C = tf.unstack(tf.shape(H))
        H_reshape = tf.reshape(H, (B * T, N, C))
        out = self.mha(H_reshape, H_reshape, training=training)
        out = tf.reshape(out, (B, T, N, self.D))
        return self.fc(out, training=training)

class TemporalAttention(Layer):
    """使用 MultiHeadAttention，对时间维做自注意力，可选因果 mask。"""
    def __init__(self, K, d, bn=False, ln=False, mask=True, attn_drop=0.0):
        super(TemporalAttention, self).__init__()
        self.K = K; self.d = d; self.D = K * d; self.mask = mask
        self.mha = MultiHeadAttention(num_heads=K, key_dim=d, attention_axes=(1,), output_shape=self.D, dropout=attn_drop)
        self.fc = DenseBlock([self.D, self.D], [tf.nn.relu, None], bn=bn, ln=ln)
    def call(self, X, STE, training=None):
        H = tf.concat((X, STE), axis=-1)  # (B, T, N, C)
        B, T, N, C = tf.unstack(tf.shape(H))
        H_reshape = tf.reshape(H, (B * N, T, C))
        out = self.mha(H_reshape, H_reshape, training=training, use_causal_mask=self.mask)
        out = tf.reshape(out, (B, T, N, self.D))
        return self.fc(out, training=training)

class GatedFusion(Layer):
    def __init__(self, D, bn=False, ln=False):
        super(GatedFusion, self).__init__()
        self.xs = DenseBlock(D, None, bn=bn, ln=ln)
        self.xt = DenseBlock(D, None, bn=bn, ln=ln)
        self.h = DenseBlock([D, D], [tf.nn.relu, None], bn=bn, ln=ln)
    def call(self, HS, HT, training=None):
        z = tf.nn.sigmoid(tf.add(self.xs(HS, training=training), self.xt(HT, training=training)))
        return self.h(tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT)), training=training)

class STAttBlock(Layer):
    def __init__(self, K, d, bn=False, ln=False, attn_drop=0.0, block_drop=0.0):
        super(STAttBlock, self).__init__()
        self.block_drop = block_drop
        self.s = SpatialAttention(K, d, bn, ln, attn_drop=attn_drop)
        self.t = TemporalAttention(K, d, bn, ln, attn_drop=attn_drop)
        self.f = GatedFusion(K * d, bn, ln)
    def call(self, X, STE, training=None):
        fused = self.f(self.s(X, STE, training=training), self.t(X, STE, training=training), training=training)
        if training and self.block_drop > 0.0:
            keep = tf.random.uniform([], 0, 1) >= self.block_drop
            scale = 1.0 / (1.0 - self.block_drop)
            return tf.cond(keep, lambda: tf.add(X, fused * scale), lambda: X)
        return tf.add(X, fused)

class TransformAttention(Layer):
    def __init__(self, K, d, bn=False, ln=False):
        super(TransformAttention, self).__init__()
        self.K, self.d, self.D = K, d, K*d
        self.query = DenseBlock(self.D, tf.nn.relu, bn=bn, ln=ln)
        self.key = DenseBlock(self.D, tf.nn.relu, bn=bn, ln=ln)
        self.value = DenseBlock(self.D, tf.nn.relu, bn=bn, ln=ln)
        self.fc = DenseBlock([self.D, self.D], [tf.nn.relu, None], bn=bn, ln=ln)
    def call(self, X, STE_P, STE_Q, training=None):
        Q = tf.transpose(tf.concat(tf.split(self.query(STE_Q, training=training), self.K, axis=-1), axis=0), (0, 2, 1, 3))
        K = tf.transpose(tf.concat(tf.split(self.key(STE_P, training=training), self.K, axis=-1), axis=0), (0, 2, 3, 1))
        V = tf.transpose(tf.concat(tf.split(self.value(X, training=training), self.K, axis=-1), axis=0), (0, 2, 1, 3))
        att = tf.nn.softmax(tf.matmul(Q, K) / (self.d ** 0.5), axis=-1)
        out = tf.transpose(tf.matmul(att, V), (0, 2, 1, 3))
        return self.fc(tf.concat(tf.split(out, self.K, 0), -1), training=training)

# =============================================================================
#  [核心修改] GMAN 主模型
# =============================================================================
class GMAN(keras.Model):
    def __init__(self, args, SE, mean, std, bn=False, **kwargs):
        super(GMAN, self).__init__(**kwargs)
        self.L = args.L; self.P = args.P; self.Q = args.Q; self.T = 24 * 60 // args.time_slot
        D = args.K * args.d
        
        # [Task 6.1] 空间异质性：将 SE 设为可训练变量 (Trainable Variable)
        # 允许模型自适应调整节点的空间表示
        self.SE = tf.Variable(SE, dtype=tf.float32, trainable=True)
        
        # 保存全局均值方差，用于还原输入数据的物理量级
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)
        
        # [Task 6.2] 时间非平稳性：初始化 RevIN
        # num_nodes=SE.shape[0] 用于初始化节点特异性参数
        self.revin = RevIN(num_nodes=SE.shape[0], num_features=1)

        attn_drop = getattr(args, 'attn_dropout', 0.0)
        block_drop = getattr(args, 'block_dropout', 0.0)

        ln = True if getattr(args, 'use_ln', False) else False

        self.fc_x = DenseBlock([D, D], [tf.nn.relu, None], bn=bn, ln=ln)
        self.st_embedding = STEmbedding(D, bn=bn, ln=ln)
        self.encoder = [STAttBlock(args.K, args.d, bn=bn, ln=ln, attn_drop=attn_drop, block_drop=block_drop) for _ in range(self.L)]
        self.transform_attention = TransformAttention(args.K, args.d, bn=bn, ln=ln)
        self.decoder = [STAttBlock(args.K, args.d, bn=bn, ln=ln, attn_drop=attn_drop, block_drop=block_drop) for _ in range(self.L)]
        self.fc_out = DenseBlock([D, 1], [tf.nn.relu, None], drop=0.1, bn=bn, ln=ln)

    def call(self, inputs, training=None):
        X, TE = inputs
        X = tf.expand_dims(X, axis=-1)
        # 对齐 dtype（混合精度下输入为 float16，常数为 float32）
        x_dtype = X.dtype
        mean = tf.cast(self.mean, x_dtype)
        std = tf.cast(self.std, x_dtype)

        # [Task 6] RevIN 逻辑链
        # 1. 恢复原始物理量级：因为输入被全局归一化过，RevIN 需要处理非平稳的 Raw Data
        X_raw = X * std + mean
        
        # 2. RevIN Normalize: 消除时间分布偏移 & 利用仿射参数对齐空间异质性
        # 这里会调用我们修复过的 normalize 方法
        X_norm, i_mean, i_std = self.revin.normalize(X_raw)

        # 3. GMAN 主体计算
        X_emb = self.fc_x(X_norm, training=training)
        STE = self.st_embedding(self.SE, TE, T=self.T, training=training)
        
        STE_P = STE[:, :self.P]; STE_Q = STE[:, self.P:]
        for block in self.encoder: X_emb = block(X_emb, STE_P, training=training)
        H = self.transform_attention(X_emb, STE_P, STE_Q, training=training)
        for block in self.decoder: H = block(H, STE_Q, training=training)
        Y_pred_norm = self.fc_out(H, training=training)

        # 4. RevIN Denormalize: 恢复相对趋势
        Y_pred_raw = self.revin.denormalize(Y_pred_norm, i_mean, i_std)

        # 直接返回真实值 (Raw Values)，不需要再做全局反归一化
        return tf.squeeze(Y_pred_raw, axis=3)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # call() 现在直接返回真实物理量级 (Raw Value)
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        for metric in self.metrics:
            if metric.name == "loss": metric.update_state(loss)
            else: metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        for metric in self.metrics:
            if metric.name == "loss": metric.update_state(loss)
            else: metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        x = data[0]
        y_pred = self(x, training=False)
        return y_pred