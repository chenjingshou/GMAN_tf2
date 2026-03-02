# dish_gman_traffic.py
import os
import argparse
import numpy as np
import tensorflow as tf
from keras.layers import (
    Layer,
    Dropout,
    LayerNormalization,
    Dense,
    MultiHeadAttention,
    GRU,
)
import keras
from keras import mixed_precision
from utils import load_data, log_string, metric


# ---------------------------------------------------------------------------
# 0. 基础设置与损失函数
# ---------------------------------------------------------------------------
mixed_precision.set_global_policy("mixed_float16")


def augment_fn(inputs, y):
    X, TE, X_frechet = inputs
    X = tf.cast(X, tf.float32)

    # -----------------------------------------------------------
    # 1. 随机遮挡 (Masking) - 真正符合物理场景的增强
    # -----------------------------------------------------------
    # 模拟传感器丢包。以 10% 的概率将输入中的点置为 0 (METR-LA中0代表缺失)
    # 这迫使 DishTS 和 GMAN 学会处理缺失值，增强图推断能力。
    if args.aug_enable:
        # 生成一个与 X 形状相同的随机掩码 (Batch, Time, Nodes, 1)
        mask_prob = 0.1  # 推荐 5% - 15%
        random_mask = tf.random.uniform(tf.shape(X)) > mask_prob

        # 将被 mask 的地方变为 0 (假设数据已经归一化，
        # 如果归一化后的0有特殊含义，这里需要设为归一化后的0值，通常是Z-score的0即均值)
        # 但在 METR-LA 原始处理中，0 是缺失值。
        # 如果 X 已经被归一化，直接乘 mask 会把点变成 0 (即均值)。
        # 这也是合理的，相当于用平均值填充缺失，让模型去修正它。
        X = X * tf.cast(random_mask, tf.float32)

    # -----------------------------------------------------------
    # 2. 极微量的物理抖动 (可选，但建议数值非常小)
    # -----------------------------------------------------------
    # 如果一定要加噪声，请先恢复到物理量级，加极小的噪声，再变回来
    # 只有在防止过拟合极其严重时才开启
    if args.aug_enable and args.aug_noise_std > 0:  # 建议设为 0 或 极小(e.g., 0.01)
        X_raw = X * std_tf + mean_tf
        # 限制噪声幅度，例如只加 +/- 2mph 的抖动，不要太大
        noise = tf.random.normal(tf.shape(X_raw), stddev=1.0)  # 1.0 mph std
        X_raw = X_raw + noise
        X = (X_raw - mean_tf) / std_tf

    # -----------------------------------------------------------
    # 3. 坚决删除 Random Walk / Drift
    # -----------------------------------------------------------
    # if args.aug_rw_std > 0: ... (DELETE THIS)

    return (X, TE, X_frechet), y


class MaskedMAELoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0), y_pred.dtype)
        mask /= tf.maximum(tf.reduce_mean(mask), 1e-6)
        loss = tf.abs(y_pred - y_true) * mask
        return tf.reduce_mean(loss)


# ---------------------------------------------------------------------------
# 1. GMAN 基础组件 (从你之前的代码中引入并固化)
# ---------------------------------------------------------------------------
class DenseBlock(Layer):
    def __init__(self, units, activations, drop=None, ln=False):
        super().__init__()
        if isinstance(units, int):
            units, activations = [units], [activations]
        self.denses = [Dense(u, kernel_initializer="glorot_uniform") for u in units]
        self.norms = [
            LayerNormalization() if ln and act else None for act in activations
        ]
        self.activations = [act for act in activations if act]
        self.dropout = Dropout(drop) if drop else None

    def call(self, x, training=None):
        for i in range(len(self.denses)):
            if self.dropout:
                x = self.dropout(x, training=training)
            x = self.denses[i](x)
            if self.norms[i]:
                x = self.norms[i](x, training=training)
            if i < len(self.activations):
                x = self.activations[i](x)
        return x


class STEmbedding(Layer):
    def __init__(self, D, T, ln=False):
        super().__init__()
        self.T = T
        self.fc_se = DenseBlock([D, D], [tf.nn.relu, None], ln=ln)
        self.fc_te = DenseBlock([D, D], [tf.nn.relu, None], ln=ln)

    def call(self, SE, TE, training=None):
        SE = tf.expand_dims(tf.expand_dims(SE, axis=0), axis=0)
        SE = self.fc_se(SE, training=training)
        dayofweek = tf.one_hot(TE[..., 0], depth=7)
        timeofday = tf.one_hot(TE[..., 1], depth=self.T)
        TE = tf.concat((dayofweek, timeofday), axis=-1)
        TE = self.fc_te(tf.expand_dims(TE, axis=2), training=training)
        return SE + TE


class SpatialAttention(Layer):
    def __init__(self, K, d, ln=False, attn_drop=0.0):
        super().__init__()
        self.D = K * d
        self.mha = MultiHeadAttention(
            num_heads=K, key_dim=d, output_shape=self.D, dropout=attn_drop
        )
        self.fc = DenseBlock([self.D, self.D], [tf.nn.relu, None], ln=ln)

    def call(self, X, STE, training=None):
        H = tf.concat((X, STE), axis=-1)
        B, T, N, C = tf.unstack(tf.shape(H))
        H_reshape = tf.reshape(H, (B * T, N, C))
        out = self.mha(H_reshape, H_reshape, training=training)
        out = tf.reshape(out, (B, T, N, self.D))
        return self.fc(out, training=training)


class TemporalAttention(Layer):
    def __init__(self, K, d, ln=False, mask=True, attn_drop=0.0):
        super().__init__()
        self.D = K * d
        self.mask = mask
        self.mha = MultiHeadAttention(
            num_heads=K, key_dim=d, output_shape=self.D, dropout=attn_drop
        )
        self.fc = DenseBlock([self.D, self.D], [tf.nn.relu, None], ln=ln)

    def call(self, X, STE, training=None):
        H = tf.concat((X, STE), axis=-1)
        B, T, N, C = tf.unstack(tf.shape(H))
        H_reshape = tf.reshape(H, (B * N, T, C))
        out = self.mha(
            H_reshape, H_reshape, training=training, use_causal_mask=self.mask
        )
        out = tf.reshape(out, (B, T, N, self.D))
        return self.fc(out, training=training)


class GatedFusion(Layer):
    def __init__(self, D, ln=False):
        super().__init__()
        self.xs = DenseBlock(D, None, ln=ln)
        self.xt = DenseBlock(D, None, ln=ln)

    def call(self, HS, HT, training=None):
        z = tf.nn.sigmoid(
            self.xs(HS, training=training) + self.xt(HT, training=training)
        )
        return (z * HS) + ((1 - z) * HT)


class STAttBlock(Layer):
    def __init__(self, K, d, ln=False, attn_drop=0.0, block_drop=0.0):
        super().__init__()
        self.block_drop_rate = block_drop
        self.s_attn = SpatialAttention(K, d, ln, attn_drop=attn_drop)
        self.t_attn = TemporalAttention(K, d, ln, attn_drop=attn_drop)
        self.fusion = GatedFusion(K * d, ln)
        self.ln = LayerNormalization() if ln else None

    def call(self, X, STE, training=None):
        HS = self.s_attn(X, STE, training=training)
        HT = self.t_attn(X, STE, training=training)
        H = self.fusion(HS, HT, training=training)
        if training and self.block_drop_rate > 0.0:
            # Stochastic depth: drop entire residual branch per sample.
            survival_rate = 1.0 - self.block_drop_rate
            batch = tf.shape(H)[0]
            mask = tf.random.uniform([batch, 1, 1, 1], 0, 1) < survival_rate
            H = tf.where(mask, H / survival_rate, tf.zeros_like(H))
        res = X + H
        if self.ln:
            res = self.ln(res, training=training)
        return res


class TransformAttention(Layer):
    def __init__(self, K, d, ln=False):
        super().__init__()
        self.K, self.d, self.D = K, d, K * d
        self.query = DenseBlock(self.D, tf.nn.relu, ln=ln)
        self.key = DenseBlock(self.D, tf.nn.relu, ln=ln)
        self.value = DenseBlock(self.D, tf.nn.relu, ln=ln)
        self.fc = DenseBlock([self.D, self.D], [tf.nn.relu, None], ln=ln)

    def call(self, X, STE_P, STE_Q, training=None):
        Q = self.query(STE_Q, training=training)
        K = self.key(STE_P, training=training)
        V = self.value(X, training=training)
        querys = tf.split(Q, self.K, axis=-1)
        keys = tf.split(K, self.K, axis=-1)
        values = tf.split(V, self.K, axis=-1)

        head_outs = []
        for i in range(self.K):
            att = tf.nn.softmax(
                tf.einsum("btnd,bsnd->bnts", querys[i], keys[i]) / (self.d**0.5)
            )
            head_outs.append(tf.einsum("bnts,bsnd->btnd", att, values[i]))

        out = tf.concat(head_outs, axis=-1)
        return self.fc(out, training=training)


# ---------------------------------------------------------------------------
# 2. 核心模型重构 (DishTS-GMAN)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2. 核心模型重构 (DishTS-GMAN) - 已修复
# ---------------------------------------------------------------------------
class CONET(Layer):
    """可学习的分布估计器，使用GRU从序列中学习分布参数。"""

    def __init__(self, num_nodes, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.gru = GRU(hidden_dim, activation="relu")
        # Output only per-node mu and sigma (2 values), not per-graph.
        self.fc = Dense(2)
        self.ln = LayerNormalization()

    def call(self, x, training=None):
        # x shape: (Batch, Time, Nodes, Features=1)
        B, T, N, F = tf.unstack(tf.shape(x))
        # (B, T, N, F) -> (B*N, T, F) for batch processing in GRU
        x_reshaped = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), (B * N, T, F))

        hidden = self.gru(x_reshaped, training=training)
        params = self.fc(hidden)
        params = tf.reshape(params, (B, N, 2))
        params = self.ln(params, training=training)

        mu, log_sigma = tf.split(params, 2, axis=-1)  # mu/log_sigma shape: (B, N, 1)
        sigma = tf.nn.softplus(log_sigma) + 1e-6

        # [修复] tf.expand_dims 一次只接受一个axis。
        # (B, N, 1) -> (B, 1, N, 1) 以便和 (B, T, N, 1) 进行广播
        return tf.expand_dims(mu, axis=1), tf.expand_dims(sigma, axis=1)


class GMAN_Core(keras.Model):
    # ... (这个类没有问题，保持原样)
    def __init__(self, args, SE, **kwargs):
        super().__init__(**kwargs)
        self.P, self.Q, self.T = args.P, args.Q, 24 * 60 // args.time_slot
        self.D = args.K * args.d
        self.SE = tf.Variable(SE, dtype=tf.float32, trainable=True)
        ln = getattr(args, "use_ln", True)

        self.fc_x = DenseBlock([self.D, self.D], [tf.nn.relu, None], ln=ln)
        self.st_embedding = STEmbedding(self.D, self.T, ln=ln)
        self.encoder = [
            STAttBlock(
                args.K,
                args.d,
                ln=ln,
                attn_drop=args.attn_dropout,
                block_drop=args.block_dropout,
            )
            for _ in range(args.L)
        ]
        self.transform_attention = TransformAttention(args.K, args.d, ln=ln)
        self.decoder = [
            STAttBlock(
                args.K,
                args.d,
                ln=ln,
                attn_drop=args.attn_dropout,
                block_drop=args.block_dropout,
            )
            for _ in range(args.L)
        ]
        self.fc_out = DenseBlock([self.D, 1], [tf.nn.relu, None], drop=0.1, ln=ln)

    def call(self, X_norm, TE, training=None):
        STE = self.st_embedding(self.SE, TE, training=training)
        STE_P, STE_Q = STE[:, : self.P], STE[:, self.P :]

        X_emb = self.fc_x(X_norm, training=training)

        enc_state = X_emb
        for block in self.encoder:
            enc_state = block(enc_state, STE_P, training=training)

        transformed_state = self.transform_attention(
            enc_state, STE_P, STE_Q, training=training
        )

        dec_state = transformed_state
        for block in self.decoder:
            dec_state = block(dec_state, STE_Q, training=training)

        Y_pred_norm = self.fc_out(dec_state, training=training)
        return Y_pred_norm, transformed_state


class DishTS_GMAN(keras.Model):
    """深度融合的预测器，整合了GMAN_Core和DishTS思想。"""

    def __init__(self, args, SE, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)
        self.num_nodes = SE.shape[0]
        conet_hidden = args.K * args.d

        self.gman_core = GMAN_Core(args, SE)
        self.back_conet = CONET(self.num_nodes, conet_hidden)

        self.hori_conet_gru = GRU(conet_hidden, activation="relu")
        # Predict mu/sigma per node for horizon distribution.
        self.hori_conet_fc = Dense(2)
        self.hori_conet_ln = LayerNormalization()

    def call(self, inputs, training=None):
        X, TE = inputs
        X = tf.expand_dims(X, axis=-1)

        X_raw = X * tf.cast(self.std, X.dtype) + tf.cast(self.mean, X.dtype)
        mu_b, sigma_b = self.back_conet(X_raw, training=training)
        X_norm = (X_raw - mu_b) / sigma_b

        Y_pred_norm, encoded_state = self.gman_core(X_norm, TE, training=training)

        B = tf.shape(encoded_state)[0]
        encoded_state_reshaped = tf.reshape(
            tf.transpose(encoded_state, [0, 2, 1, 3]),
            (B * self.num_nodes, self.gman_core.Q, self.gman_core.D),
        )

        future_dist_hidden = self.hori_conet_gru(
            encoded_state_reshaped, training=training
        )
        params_hori = self.hori_conet_fc(future_dist_hidden)
        params_hori = tf.reshape(params_hori, (B, self.num_nodes, 2))
        params_hori = self.hori_conet_ln(params_hori, training=training)

        mu_h, log_sigma_h = tf.split(
            params_hori, 2, axis=-1
        )  # mu_h/sigma_h shape: (B, N, 1)
        sigma_h = tf.nn.softplus(log_sigma_h) + 1e-6

        mu_h = tf.expand_dims(mu_h, axis=1)
        sigma_h = tf.expand_dims(sigma_h, axis=1)

        Y_pred_raw = Y_pred_norm * sigma_h + mu_h
        return tf.squeeze(Y_pred_raw, axis=3)


class FusionLayer(Layer):
    """融合 Fréchet 全局指纹与节点级时空摘要的门控层。"""

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = Dense(hidden_dim, activation="relu")
        self.dense2 = Dense(hidden_dim, activation="relu")
        self.gate = Dense(hidden_dim, activation="sigmoid")
        self.ln = LayerNormalization()

    def call(self, x1, x2, training=None):
        h1 = self.dense1(x1)
        h2 = self.dense2(x2)
        g = self.gate(tf.concat([x1, x2], axis=-1))
        fused = g * h1 + (1 - g) * h2
        return self.ln(fused + x1, training=training)


class DishTS_GMAN_Plus(keras.Model):
    """融合 Fréchet embedding 的 GMAN 变体。"""

    def __init__(self, args, SE, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)
        self.num_nodes = SE.shape[0]
        self.frechet_dim = getattr(args, "frechet_dim", 16)
        conet_hidden = args.K * args.d

        self.gman_core = GMAN_Core(args, SE)
        self.back_conet = CONET(self.num_nodes, conet_hidden)

        # Fréchet embedding projection + fusion
        self.frechet_projector = Dense(conet_hidden, activation="relu")
        self.fusion = FusionLayer(conet_hidden)
        self.fused_ln = LayerNormalization()

        # Horizon distribution head
        self.hori_conet_gru = GRU(conet_hidden, activation="relu")
        self.hori_conet_fc = Dense(2)
        self.hori_conet_ln = LayerNormalization()

    def call(self, inputs, training=None):
        X, TE, X_frechet = inputs
        X = tf.expand_dims(X, axis=-1)

        X_raw = X * tf.cast(self.std, X.dtype) + tf.cast(self.mean, X.dtype)
        mu_b, sigma_b = self.back_conet(X_raw, training=training)
        X_norm = (X_raw - mu_b) / sigma_b

        Y_pred_norm, encoded_state = self.gman_core(X_norm, TE, training=training)

        # Node-level summary from encoder output
        encoded_state_summary = tf.reduce_mean(encoded_state, axis=1)  # (B, N, D)

        # Frechet embedding -> broadcast per node
        frechet_summary = self.frechet_projector(X_frechet, training=training)  # (B, D)
        frechet_summary = tf.expand_dims(frechet_summary, axis=1)
        frechet_summary = tf.broadcast_to(
            frechet_summary,
            [tf.shape(encoded_state_summary)[0], self.num_nodes, tf.shape(frechet_summary)[-1]],
        )

        fused_summary = self.fusion(
            encoded_state_summary, frechet_summary, training=training
        )  # (B, N, D)
        fused_summary = self.fused_ln(fused_summary, training=training)

        # Horizon distribution conditioned on fused summary
        B = tf.shape(fused_summary)[0]
        fused_reshaped = tf.reshape(
            fused_summary, (B * self.num_nodes, 1, self.gman_core.D)
        )
        future_dist_hidden = self.hori_conet_gru(
            fused_reshaped, training=training
        )  # (B*N, D)

        params_hori = self.hori_conet_fc(future_dist_hidden)
        params_hori = tf.reshape(params_hori, (B, self.num_nodes, 2))
        params_hori = self.hori_conet_ln(params_hori, training=training)

        mu_h, log_sigma_h = tf.split(params_hori, 2, axis=-1)
        sigma_h = tf.nn.softplus(log_sigma_h) + 1e-6

        mu_h = tf.expand_dims(mu_h, axis=1)
        sigma_h = tf.expand_dims(sigma_h, axis=1)

        Y_pred_raw = Y_pred_norm * sigma_h + mu_h
        return tf.squeeze(Y_pred_raw, axis=3)

# ---------------------------------------------------------------------------
# 3. 训练与评估工作流
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 基础参数 (保持不变)
    parser.add_argument("--time_slot", type=int, default=5)
    parser.add_argument("--P", type=int, default=12, help="history steps")
    parser.add_argument("--Q", type=int, default=12, help="prediction steps")
    parser.add_argument("--L", type=int, default=5, help="number of STAtt Blocks")
    parser.add_argument("--K", type=int, default=8, help="number of attention heads")
    parser.add_argument("--d", type=int, default=8, help="dims of each head")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=100)  # 减少epoch以便快速演示
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--decay_epoch", type=int, default=10)
    parser.add_argument("--frechet_dim", type=int, default=16, help="Fréchet embedding dimension")
    parser.add_argument("--frechet_file", default=None, help="Path to precomputed Fréchet embeddings (.npy)")

    # 正则化与模型结构参数
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument(
        "--block_dropout", type=float, default=0.1, help="stochastic depth drop rate"
    )
    parser.add_argument("--use_ln", default=True, type=bool)
    parser.add_argument("--aug_enable", default=True, type=bool, help="enable mild data augmentation")
    parser.add_argument("--aug_noise_std", type=float, default=0.01, help="stddev of additive Gaussian noise on raw speeds")
    parser.add_argument("--aug_rw_std", type=float, default=0.0, help="stddev of per-step random walk drift on raw speeds")

    # 文件路径
    parser.add_argument("--traffic_file", default="data/METR.h5")
    parser.add_argument("--SE_file", default="data/SE(METR).txt")
    parser.add_argument("--model_file", default="./models/DishTS_GMAN.weights.h5")
    parser.add_argument("--log_file", default="./log/DishTS_GMAN.log")
    args = parser.parse_args()

    # 创建目录
    for path in ["models", "log"]:
        if not os.path.exists(path):
            os.makedirs(path)
    log_f = open(args.log_file, "w")
    log_string(log_f, str(args))

    # 数据加载
    log_string(log_f, "Loading data...")
    (
        trainX,
        trainTE,
        trainFrechet,
        trainY,
        valX,
        valTE,
        valFrechet,
        valY,
        testX,
        testTE,
        testFrechet,
        testY,
        SE,
        mean,
        std,
    ) = load_data(args)
    log_string(log_f, f"trainX: {trainX.shape}\ttrainY: {trainY.shape}")

    mean_tf = tf.constant(mean, dtype=tf.float32)
    std_tf = tf.constant(std, dtype=tf.float32)

    # 构建 tf.data.Dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices(((trainX, trainTE, trainFrechet), trainY))
        .shuffle(2048)
        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(((valX, valTE, valFrechet), valY))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices(((testX, testTE, testFrechet), testY))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 模型编译
    log_string(log_f, "Compiling model...")
    model = DishTS_GMAN_Plus(args, SE, mean, std)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.decay_epoch * (len(trainX) // args.batch_size),
        decay_rate=0.8,
        staircase=True,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=MaskedMAELoss(),
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    # 训练模型
    log_string(log_f, "**** Training model ****")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            args.model_file, save_best_only=True, save_weights_only=True
        ),
    ]
    model.fit(
        train_ds,
        epochs=args.max_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # 测试模型
    log_string(log_f, "**** Testing model ****")
    model.load_weights(args.model_file)
    test_pred = model.predict(test_ds)

    mae, rmse, mape = metric(test_pred, testY)
    log_string(
        log_f,
        f"Final Test Results: MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape*100:.2f}%",
    )
    log_f.close()
