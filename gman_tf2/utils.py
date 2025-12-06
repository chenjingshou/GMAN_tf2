import numpy as np
import pandas as pd
import os

def log_string(log, string):
    """
    Write a string to the log file and print it to the console.
    """
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, label):
    """
    Calculate performance metrics: MAE, RMSE, MAPE.
    参照原始模板：使用mask处理0值
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        
    return mae, rmse, mape

def seq2instance(data, P, Q):
    """
    Convert sequence data into instances for training/testing.
    :param data: sequence data.
    :param P: history steps.
    :param Q: prediction steps.
    :return: x, y
    """
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y

def load_data(args):
    """
    Load and preprocess data.
    参照原始模板：使用Time.freq.delta.total_seconds()计算时间
    """
    # 验证文件存在
    if not os.path.exists(args.traffic_file):
        raise FileNotFoundError(f"Traffic file not found: {args.traffic_file}")
    if not os.path.exists(args.SE_file):
        raise FileNotFoundError(f"SE file not found: {args.SE_file}")
    
    # Traffic data
    df = pd.read_hdf(args.traffic_file)
    Traffic = df.values
    
    print(f"Loaded traffic data shape: {Traffic.shape}")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Time frequency: {df.index.freq}")

    # Split data
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    
    # 验证数据集大小
    assert train_steps > args.P + args.Q, f"Training set too small: {train_steps} steps"
    assert val_steps > args.P + args.Q, f"Validation set too small: {val_steps} steps"
    assert test_steps > args.P + args.Q, f"Test set too small: {test_steps} steps"
    
    train = Traffic[:train_steps]
    val = Traffic[train_steps: train_steps + val_steps]
    test = Traffic[-test_steps:]

    # Create instances
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)

    # Normalization - 只归一化X，不归一化Y（参照原始模板）
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std
    # trainY, valY, testY 保持原始尺度

    # Spatial Embedding
    try:
        with open(args.SE_file, mode='r') as f:
            lines = f.readlines()
            temp = lines[0].strip().split(' ')
            N, dims = int(temp[0]), int(temp[1])
            SE = np.zeros(shape=(N, dims), dtype=np.float32)
            for line in lines[1:]:
                temp = line.strip().split(' ')
                index = int(temp[0])
                SE[index] = [float(ch) for ch in temp[1:]]
        print(f"Loaded spatial embedding: {SE.shape}")
    except Exception as e:
        raise ValueError(f"Error loading SE file: {e}")

    # Temporal Embedding - 参照原始模板使用Time.freq
    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    
    # 关键修复：使用数据实际频率（参照原始模板）
    if hasattr(Time.freq, 'delta'):
        # 如果freq有delta属性，使用它
        freq_seconds = Time.freq.delta.total_seconds()
    else:
        # 降级方案：使用参数指定的时间间隔
        freq_seconds = args.time_slot * 60
        print(f"Warning: Using args.time_slot ({args.time_slot} min) as time frequency")
    
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) // freq_seconds
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)
    
    print(f"Time embedding range - day: [0, 6], time: [0, {int(timeofday.max())}]")

    # Create temporal instances
    train_te = seq2instance(Time[:train_steps], args.P, args.Q)
    train_te = np.concatenate(train_te, axis=1).astype(np.int32)
    val_te = seq2instance(Time[train_steps: train_steps + val_steps], args.P, args.Q)
    val_te = np.concatenate(val_te, axis=1).astype(np.int32)
    test_te = seq2instance(Time[-test_steps:], args.P, args.Q)
    test_te = np.concatenate(test_te, axis=1).astype(np.int32)

    return (trainX, train_te, trainY, valX, val_te, valY, testX, test_te, testY,
            SE, mean, std)