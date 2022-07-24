import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


class LstmRnn(pl.LightningModule):
    def __init__(self, n_input, n_hidden, n_classes, lr):
        super().__init__()
        self.lr = lr
        self.input_linear = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(n_hidden, n_hidden, num_layers=2, batch_first=True)
        self.output_linear = nn.Linear(n_hidden, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        x = self.input_linear(x)
        x = self.relu(x)
        output, _ = self.lstm(x)
        x = output[:, -1, :]
        x = self.output_linear(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self(x)
        loss = self.loss_fn(output, y.view(-1, ))
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        output = self(x)
        loss = self.loss_fn(output, y.view(-1, ))
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        predicts = torch.argmax(output, dim=1)
        self.val_accuracy.update(predicts, y.view(-1, ))
        self.log('val_acc', self.val_accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return loss


class CNN(pl.LightningModule):
    def __init__(self, n_input, n_steps, n_classes, lr, p_dropout=0.5):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy()
        self.lr = lr
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.conv1 = nn.Conv1d(in_channels=n_input, out_channels=n_input * 2, kernel_size=2, padding='same')
        self.conv2 = nn.Conv1d(in_channels=n_input * 2, out_channels=n_input * 4, kernel_size=2, padding='same')
        self.conv3 = nn.Conv1d(in_channels=n_input * 4, out_channels=n_input * 8, kernel_size=2, padding='same')
        self.conv4 = nn.Conv1d(in_channels=n_input * 8, out_channels=n_input * 16, kernel_size=2, padding='same')

        self.dropout = nn.Dropout(p=p_dropout)
        self.linear = nn.Linear(n_input * n_steps, n_classes)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        output = self.linear(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self(x)
        loss = self.loss_fn(output, y.view(-1, ))
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        output = self(x)
        loss = self.loss_fn(output, y.view(-1, ))
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        predicts = torch.argmax(output, dim=1)
        self.val_accuracy.update(predicts, y.view(-1, ))
        self.log('val_acc', self.val_accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return loss
