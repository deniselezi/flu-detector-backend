import numpy as np
import pandas as pd
import pickle
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.classes.FFNN import FFNN
from models.classes.RNN import GRU

model_dict = {
    "lasso": ("./models/lasso.pkl", None, "./data/final_1000.csv"),
    "ffnn": ("./models/ffnn.pt", FFNN, "./data/final_200.csv"),
    "rnn": ("./models/rnn.pt", GRU, "./data/final_300.csv"),
}


def fetch_preds(model_string, start_idx, end_idx):
    """
    Predicts values from the corresponding model between start_idx and end_idx and returns them.
    """
    model_path, model_class, data_path = model_dict[model_string]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "lasso" in model_path:  # check needed because sklearn models need pickle
        data = pd.read_csv(data_path)
        x = x_scaler.fit_transform(data)
        x = x[start_idx:end_idx]

        ili = pd.read_csv("data/final_ili.csv", header=None)
        y = y_scaler.fit_transform(ili)
        y = y[start_idx:end_idx]

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y_pred_scaled = model.predict(x)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_actual = y_scaler.inverse_transform(y.reshape(-1, 1))

    elif "rnn" in model_path:
        window_size = 28

        data = pd.read_csv(data_path)
        ili = pd.read_csv("data/final_ili.csv", header=None)

        # create windows (for input sequences and hindcasting)
        data, ili = create_windows(data, ili)

        start_idx = start_idx - (window_size - 1)
        end_idx = end_idx - (window_size - 1)
        x = data[start_idx:end_idx]
        y = ili[start_idx:end_idx]

        # needed to fit the scalers, but not used for the predicting
        x_fit = data[:start_idx]
        y_fit = ili[:start_idx]
        n_samples, _, n_features = x_fit.shape

        x_fit_reshaped = x_fit.reshape(-1, n_features)
        x_fit = x_scaler.fit_transform(x_fit_reshaped).reshape(
            n_samples, window_size, n_features
        )
        y_fit = y_scaler.fit_transform(y_fit).reshape(n_samples, window_size)

        del x_fit
        del y_fit

        n_samples, _, n_features = x.shape

        # scale data
        x_reshaped = x.reshape(-1, n_features)
        x = x_scaler.transform(x_reshaped).reshape(n_samples, window_size, n_features)
        y = y_scaler.transform(y).reshape(n_samples, window_size)

        # convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        input_dim = n_features

        model = model_class(input_dim)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        with torch.no_grad():  # get predictions
            y_pred_tensor = model(x_tensor)

        y_pred_scaled = y_pred_tensor.cpu().numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        y_actual = y_scaler.inverse_transform(
            y_tensor.cpu().numpy()
        )  # essentially ili as a np array

        y_pred = y_pred[:, -1].reshape(-1, 1)
        y_actual = y_actual[:, -1].reshape(-1, 1)

    else:  # ffnn
        data = pd.read_csv(data_path)
        x = x_scaler.fit_transform(data)
        x = x[start_idx:end_idx]

        ili = pd.read_csv("data/final_ili.csv", header=None)
        y = y_scaler.fit_transform(ili)
        y = y[start_idx:end_idx]

        # convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        input_dim = x_tensor.shape[1]

        model = model_class(input_dim)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        with torch.no_grad():  # get predictions
            y_pred_tensor = model(x_tensor)

        y_pred_scaled = y_pred_tensor.cpu().numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        y_actual = y_scaler.inverse_transform(
            y_tensor.cpu().numpy()
        )  # essentially ili as a np array

    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    correlation = np.corrcoef(y_actual.flatten(), y_pred.flatten())[0, 1]

    print(mae, mse, correlation)

    return y_pred.tolist(), y_actual.tolist()


def create_windows(x, y, window_size=28):
    x_windows = []
    y_windows = []

    x, y = x.values, y.values

    n_samples = len(x)

    x_windows = np.array(
        [x[i : i + window_size] for i in range(n_samples - window_size + 1)]
    )
    y_windows = np.array(
        [y[i : i + window_size].flatten() for i in range(n_samples - window_size + 1)]
    )

    return x_windows, y_windows
