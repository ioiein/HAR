import click
from torch.utils import data
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from models import LstmRnn, CNN
from data import make_dataset


@click.command()
@click.option('-n', '--name', required=True, help='name of experiment')
@click.option('-e', '--epochs', type=int, default=100, help='count of training epochs')
@click.option('-b', '--batch-size', type=int, default=16, help='batch size')
@click.option('--gpus', type=int, default=0, help='count of GPU')
@click.option('-d', '--dataset_path', required=True, help='path to dataset')
@click.option('--lstm/--cnn', required=True, is_flag=True)
@click.option('--n_input', type=int, default=9, help='count of input channels')
@click.option('--n_series', type=int, default=128, help='length of input series')
@click.option('--n_hidden', type=int, default=64, help='size of hidden layer for recurrent network')
@click.option('-c', '--n_class', type=int, required=True, help='count of classes')
@click.option('-l', '--learning_rate', type=float, default=0.0025, help='training learning rate')
def train(name, epochs, batch_size, gpus, dataset_path, lstm, n_input, n_series, n_hidden, n_class, learning_rate):
    with open('input_signal_types.txt') as f:
        input_signals = f.read().splitlines()
    train_dataset, test_dataset = make_dataset(input_signals, dataset_path)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,
                                       num_workers=2)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                                      num_workers=2)
    logger = TensorBoardLogger('tb_logs', name=name)
    if lstm:
        model = LstmRnn(n_input, n_hidden, n_class, lr=learning_rate)
    else:
        model = CNN(n_input, n_series, n_class, lr=learning_rate)
    if gpus:
        accelerator = 'gpu'
        devices = gpus
    else:
        accelerator = 'cpu'
        devices = 1
    trainer = Trainer(accelerator=accelerator, devices=devices, logger=logger, max_epochs=epochs)
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    train()
