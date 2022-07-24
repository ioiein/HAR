# <a title="Activity Recognition" > Human Activity Recognition</a>

Released two models: RNN LSTM and CNN for classifying human activity by sensors like accelerometers and gyroscopes.

Model learning provided with PyTorch Lightning. For logging used TensorboardLogger, so you can use it to analise learning process. 

## Data

For tests was used [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones). You can download it by [direct link](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip)

Accuracy score on this dataset:

| Model    | Accuracy |
|----------|----------|
| RNN LSTM | 0.91     |
| CNN      | 0.92     |

## Usage

```
 python src\train.py -n name -d path_for_dataset -c number_of_classes --lstn/cnn
```

For detailed options use 

```
python src\train.py --help
```