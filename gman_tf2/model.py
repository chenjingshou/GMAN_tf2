import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, BatchNormalization

class FC(Layer):
    """严格按照原版逻辑实现"""
    def __init__(self, units, activations, use_bias=True, drop=None, bn=False):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            activations = list(activations)
        
        self.bn = bn
        self.drop = drop
        self.conv_layers = []
        self.bn_layers = []
        
        for num_unit, activation in zip(units, activations):
            # Conv层：按传入的use_bias决定
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=num_unit,
                    kernel_size=[1, 1],
                    strides=[1, 1],
                    padding='VALID',
                    use_bias=use_bias,
                    activation=None,
                    kernel_initializer='glorot_uniform'
                )
            )
            
            # BN层（如果需要且有activation）
            if activation is not None and bn:
                self.bn_layers.append(BatchNormalization(momentum=0.9))
            else:
                self.bn_layers.append(None)
                
        self.activations = activations
        
        if self.drop is not None:
            self.dropout_layer = Dropout(drop)

    def call(self, x, training=None):
        for conv, bn_layer, activation in zip(self.conv_layers, self.bn_layers, self.activations):
            if self.drop is not None:
                x = self.dropout_layer(x, training=training)
            
            x = conv(x)
            
            if activation is not None:
                if bn_layer is not None:
                    x = bn_layer(x, training=training)
                x = activation(x)
        
        return x


class MaskedMAELoss(tf.keras.losses.Loss):
    """修复：添加mask处理的MAE损失"""
    def call(self, y_true, y_pred):
        compute_dtype = y_pred.dtype
        
        mask = tf.not_equal(y_true, 0)
        mask = tf.cast(mask, compute_dtype)
        
        mask_mean = tf.reduce_mean(mask)
        mask = tf.where(
            tf.math.is_finite(mask / mask_mean),
            mask / mask_mean,
            tf.zeros_like(mask)
        )
        
        mae = tf.abs(y_pred - y_true)
        masked_mae = mae * mask
        
        masked_mae = tf.where(
            tf.math.is_finite(masked_mae),
            masked_mae,
            tf.zeros_like(masked_mae)
        )
        
        return tf.cast(tf.reduce_mean(masked_mae), tf.float32)


class STEmbedding(Layer):
    def __init__(self, D, bn=False):
        super(STEmbedding, self).__init__()
        self.D = D
        self.fc_se = FC(units=[D, D], activations=[tf.nn.relu, None], bn=bn)
        self.fc_te = FC(units=[D, D], activations=[tf.nn.relu, None], bn=bn)

    def call(self, SE, TE, T, training=None):
        SE = tf.expand_dims(tf.expand_dims(SE, axis=0), axis=0)
        SE = self.fc_se(SE, training=training)
        
        dayofweek = tf.one_hot(TE[..., 0], depth=7)
        timeofday = tf.one_hot(TE[..., 1], depth=T)
        TE = tf.concat((dayofweek, timeofday), axis=-1)
        TE = tf.expand_dims(TE, axis=2)
        TE = self.fc_te(TE, training=training)
        return tf.add(SE, TE)


class SpatialAttention(Layer):
    def __init__(self, K, d, bn=False):  # ✅ 统一用bn
        super(SpatialAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K * d
        self.query = FC(self.D, tf.nn.relu, bn=bn)  # ✅ 传递bn
        self.key = FC(self.D, tf.nn.relu, bn=bn)
        self.value = FC(self.D, tf.nn.relu, bn=bn)
        self.fc = FC([self.D, self.D], [tf.nn.relu, None], bn=bn)

    def call(self, X, STE, training=None):
        X_ = tf.concat((X, STE), axis=-1)
        query = self.query(X_, training=training)
        key = self.key(X_, training=training)
        value = self.value(X_, training=training)

        query = tf.concat(tf.split(query, self.K, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.K, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.K, axis=-1), axis=0)

        attention = tf.matmul(query, key, transpose_b=True)
        attention /= (self.d ** 0.5)
        attention = tf.nn.softmax(attention, axis=-1)

        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, self.K, axis=0), axis=-1)
        X = self.fc(X, training=training)
        return X


class TemporalAttention(Layer):
    def __init__(self, K, d, bn=False, mask=True):  # ✅ 统一用bn
        super(TemporalAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K * d
        self.mask = mask
        self.query = FC(self.D, tf.nn.relu, bn=bn)  # ✅ 传递bn
        self.key = FC(self.D, tf.nn.relu, bn=bn)
        self.value = FC(self.D, tf.nn.relu, bn=bn)
        self.fc = FC([self.D, self.D], [tf.nn.relu, None], bn=bn)

    def call(self, X, STE, training=None):
        X_ = tf.concat((X, STE), axis=-1)
        query = self.query(X_, training=training)
        key = self.key(X_, training=training)
        value = self.value(X_, training=training)

        query = tf.concat(tf.split(query, self.K, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.K, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.K, axis=-1), axis=0)

        query = tf.transpose(query, perm=(0, 2, 1, 3))
        key = tf.transpose(key, perm=(0, 2, 3, 1))
        value = tf.transpose(value, perm=(0, 2, 1, 3))

        attention = tf.matmul(query, key)
        attention /= (self.d ** 0.5)

        if self.mask:
            num_step = tf.shape(query)[2]
            mask = tf.linalg.band_part(tf.ones((num_step, num_step)), -1, 0)
            mask = tf.cast(mask, dtype=tf.bool)
            attention = tf.where(mask, attention, -1e9)

        attention = tf.nn.softmax(attention, axis=-1)

        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        X = tf.concat(tf.split(X, self.K, axis=0), axis=-1)
        X = self.fc(X, training=training)
        return X


class GatedFusion(Layer):
    def __init__(self, D, bn=False):
        super(GatedFusion, self).__init__()
        self.D = D
        self.fc_xs = FC(D, None, use_bias=False, bn=bn)
        self.fc_xt = FC(D, None, use_bias=True, bn=bn)
        self.fc_h = FC([D, D], [tf.nn.relu, None], bn=bn)

    def call(self, HS, HT, training=None):
        XS = self.fc_xs(HS, training=training)
        XT = self.fc_xt(HT, training=training)
        z = tf.nn.sigmoid(tf.add(XS, XT))
        H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
        H = self.fc_h(H, training=training)
        return H


class STAttBlock(Layer):
    def __init__(self, K, d, bn=False):  # ✅ 统一用bn
        super(STAttBlock, self).__init__()
        self.spatial_attention = SpatialAttention(K, d, bn=bn)  # ✅ 传递bn
        self.temporal_attention = TemporalAttention(K, d, bn=bn)
        self.gated_fusion = GatedFusion(K * d, bn=bn)

    def call(self, X, STE, training=None):
        HS = self.spatial_attention(X, STE, training=training)
        HT = self.temporal_attention(X, STE, training=training)
        H = self.gated_fusion(HS, HT, training=training)
        return tf.add(X, H)


class TransformAttention(Layer):
    def __init__(self, K, d, bn=False):  # ✅ 统一用bn
        super(TransformAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K * d
        self.query = FC(self.D, tf.nn.relu, bn=bn)  # ✅ 传递bn
        self.key = FC(self.D, tf.nn.relu, bn=bn)
        self.value = FC(self.D, tf.nn.relu, bn=bn)
        self.fc = FC([self.D, self.D], [tf.nn.relu, None], bn=bn)

    def call(self, X, STE_P, STE_Q, training=None):
        query = self.query(STE_Q, training=training)
        key = self.key(STE_P, training=training)
        value = self.value(X, training=training)

        query = tf.concat(tf.split(query, self.K, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.K, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.K, axis=-1), axis=0)

        query = tf.transpose(query, perm=(0, 2, 1, 3))
        key = tf.transpose(key, perm=(0, 2, 3, 1))
        value = tf.transpose(value, perm=(0, 2, 1, 3))

        attention = tf.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = tf.nn.softmax(attention, axis=-1)

        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        X = tf.concat(tf.split(X, self.K, axis=0), axis=-1)
        X = self.fc(X, training=training)
        return X


class GMAN(tf.keras.Model):
    def __init__(self, args, SE, mean, std, bn=False, **kwargs):
        super(GMAN, self).__init__(**kwargs)
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.T = 24 * 60 // args.time_slot
        D = args.K * args.d
        
        # ✅ 关键修复：保存这些属性
        self.SE = tf.constant(SE, dtype=tf.float32)
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)
        
        self.fc_x = FC([D, D], [tf.nn.relu, None], bn=bn)
        self.st_embedding = STEmbedding(D, bn=bn)
        self.encoder = [STAttBlock(args.K, args.d, bn=bn) for _ in range(self.L)]
        self.transform_attention = TransformAttention(args.K, args.d, bn=bn)
        self.decoder = [STAttBlock(args.K, args.d, bn=bn) for _ in range(self.L)]
        self.fc_out = FC([D, 1], [tf.nn.relu, None], drop=0.1, bn=bn)

    def call(self, inputs, training=None):
        X, TE = inputs
        X = tf.expand_dims(X, axis=-1)
        X = self.fc_x(X, training=training)

        STE = self.st_embedding(self.SE, TE, T=self.T, training=training)
        STE_P = STE[:, :self.P]
        STE_Q = STE[:, self.P:]

        for block in self.encoder:
            X = block(X, STE_P, training=training)

        X = self.transform_attention(X, STE_P, STE_Q, training=training)

        for block in self.decoder:
            X = block(X, STE_Q, training=training)

        X = self.fc_out(X, training=training)
        return tf.squeeze(X, axis=3)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # 反归一化
            y_pred = y_pred * self.std + self.mean
            loss = self.compute_loss(y=y, y_pred=y_pred)
            
            # 处理混合精度
            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)
        
        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        y_pred = y_pred * self.std + self.mean
        loss = self.compute_loss(y=y, y_pred=y_pred)
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        x = data[0]
        y_pred = self(x, training=False)
        return y_pred * self.std + self.mean