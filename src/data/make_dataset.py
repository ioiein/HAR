import numpy as np
import torch
from torch.utils import data
import os


def load_x(x_signals_paths):
    x_signals = []

    for signal_type_path in x_signals_paths:
        file = open(signal_type_path, 'r')
        x_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(x_signals), (1, 2, 0))


def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    return y_ - 1


class SeriesDataset(data.Dataset):
    def __init__(self, x, y):
        self.X = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y.type(torch.LongTensor)


def make_dataset(input_signal_types, dataset_path):
    print("\n" + "Dataset is located at: " + dataset_path)
    x_train_signal_paths = [os.path.join(*[dataset_path, "train/", "Inertial Signals/", signal + "train.txt"])
                            for signal in input_signal_types]
    x_test_signal_paths = [os.path.join(*[dataset_path, "test/", "Inertial Signals/", signal + "test.txt"])
                           for signal in input_signal_types]

    x_train = load_x(x_train_signal_paths)
    x_test = load_x(x_test_signal_paths)

    y_train = load_y(os.path.join(dataset_path, "train/", "y_train.txt"))
    y_test = load_y(os.path.join(dataset_path, "test/", "y_test.txt"))

    train_dataset = SeriesDataset(x_train, y_train)
    test_dataset = SeriesDataset(x_test, y_test)
    return train_dataset, test_dataset
