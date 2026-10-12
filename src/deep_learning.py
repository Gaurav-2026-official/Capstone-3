"""Optional TensorFlow sequence models. Install tensorflow to use these functions."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .data_utils import regression_metrics

def evaluate_neural_models(series, test_size=0.2, lookback=30, epochs=25, batch_size=32):
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as e:
        raise ImportError("Optional deep learning requires TensorFlow. Install it with: pip install tensorflow") from e
    y = pd.Series(series).dropna().astype("float32")
    n_test = max(1, int(round(len(y) * test_size)))
    split = len(y) - n_test
    scaler = MinMaxScaler()
    scaler.fit(y.iloc[:split].to_numpy().reshape(-1,1))  # no leakage: fit on train only
    scaled = scaler.transform(y.to_numpy().reshape(-1,1)).ravel()
    X, target = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i]); target.append(scaled[i])
    X, target = np.asarray(X), np.asarray(target)
    target_dates = y.index[lookback:]
    train_mask = target_dates < y.index[split]
    X_train, y_train = X[train_mask], target[train_mask]
    # chronological validation tail of training windows
    val_n = max(1, int(.1*len(X_train)))
    X_fit, y_fit = X_train[:-val_n], y_train[:-val_n]
    X_val, y_val = X_train[-val_n:], y_train[-val_n:]
    X_test = X[target_dates >= y.index[split]]
    if len(X_test) == 0: raise ValueError("Not enough observations for requested lookback/test size.")
    configs = [
        ("RNN", SimpleRNN(32)),
        ("LSTM", LSTM(32)),
        ("GRU", GRU(32)),
    ]
    preds = pd.DataFrame(index=y.index[split:])
    preds["Actual"] = y.iloc[split:].values
    rows = []
    for name, layer in configs:
        model = Sequential([Input(shape=(lookback,1)), layer, Dense(16, activation="relu"), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_fit.reshape(-1,lookback,1), y_fit, validation_data=(X_val.reshape(-1,lookback,1),y_val),
                  epochs=epochs, batch_size=batch_size, shuffle=False,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
        scaled_pred = model.predict(X_test.reshape(-1,lookback,1), verbose=0)
        pred = scaler.inverse_transform(scaled_pred).ravel()
        actual = y.iloc[split:].iloc[:len(pred)]
        preds[name] = pred[:len(actual)]
        rows.append({"Model":name, **regression_metrics(actual, pred[:len(actual)])})
    return preds, pd.DataFrame(rows).sort_values("RMSE")
