# SQL Injection Detection Using Hugging Face Transformers

This project fine-tunes the `google/mobilebert-uncased` model from Hugging Face's Transformers library to detect SQL injection attacks. The fine-tuning process is customizable, supporting Nvidia's APEX for mixed precision training and leveraging optimizers like Adafactor or AdamW. The model is trained on a custom dataset and uses techniques such as dynamic padding and learning rate scheduling.

## Features
- Uses the **MobileBERT** model (`google/mobilebert-uncased`) for sequence classification.
- Fine-tuning to detect SQL injection attacks using labeled data.
- Optimized training with **Adafactor** and **AdamW** optimizers.
- Supports mixed precision training using **NVIDIA Apex** for efficient use of GPU resources.
- Implements a dynamic padding mechanism for input sequences.
- Learning rate scheduling with a warm-up period using a linear scheduler.

## Requirements

- Python 3.7+
- `torch`
- `transformers`
- `nvidia-apex` (optional, for mixed precision training)
- `scikit-learn` (for evaluation metrics)
- `pandas`
- `csv` (if working with CSV files)

Install the dependencies using:

```bash
pip install torch transformers scikit-learn pandas
```

To install NVIDIA Apex (for mixed precision training):

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

## Configuration

You can control the model, data, and optimizer settings through the `config` dictionary. Here's a breakdown of important configuration parameters:

- `model_name`: Pre-trained model to be used. Set to `google/mobilebert-uncased`.
- `optimizer`: Choose between `Adafactor` or `AdamW`.
- `use_amp`: Enable Automatic Mixed Precision (AMP) with `True` or `False`.
- `data_file`: Path to your dataset. Expecting a CSV file with columns `Query` and `Label`.
- `EPOCHS`: Number of training epochs.
- `lr`: Learning rate.
- `scheduler`: Type of learning rate scheduler (e.g., `linear`).
- `log_batches`: Log every `N` batches during training.

Here is an example configuration:

```python
config = {
    'model': 'bertmobile-sql-inject-detector',
    'model_name': 'google/mobilebert-uncased',
    'ft_model': '',  # Path to fine-tuned model (optional)
    'ft_model_sd': '',  # Path to model state dictionary (optional)
    'optimizer': 'Adafactor',
    'lr': 1e-4,
    'weight_decay': 0.01,
    'EPOCHS': 5,
    'data_file': 'data/Modified_SQL_Dataset.csv',
    'data_format': 'csv',
    'data_keys': ['Query', 'Label'],
    'save_loc': 'modelmobile_v1',
    'log_batches': 50,
    'version': 'v1',
    'warmup_steps': 94,
    'scheduler': 'linear',
    'adafactor_config': {
        'lr': 1e-3,
        'eps': (1e-30, 1e-3),
        'clip_threshold': 1.0,
        'decay_rate': -0.8,
    },
}
```

## Data Format

The data should be a CSV file with two columns:
- `Query`: The SQL query text.
- `Label`: The label indicating whether the query is a SQL injection (1) or not (0).

## Training

You can train the model by initializing the `SQLInjectionPipeline` with the config and calling the `setup_and_train` method:

```python
pipeline = SQLInjectionPipeline(config)
results = pipeline.setup_and_train()
```

## Evaluation

During training, the model is evaluated after each epoch using precision, recall, F1-score, and ROC-AUC score. These metrics are logged, and you can access the results:

```python
for result in results:
    print(result)
```

## Model Saving

The model is saved after each epoch in the specified `save_loc` directory with the filename structure: `{model}_{version}_model_epoch_{epoch}.pt`.

## Dataset Used
Dataset used is from Kaggle . Here is the location: https://www.kaggle.com/datasets/sajid576/sql-injection-dataset  by Sajid57.

## Advanced Features

- **Mixed Precision Training**: If you have a compatible NVIDIA GPU, you can enable mixed precision training with the `use_amp` flag for faster training with reduced memory usage.
- **DataParallel**: The model supports multi-GPU training using `torch.nn.DataParallel` if more than one GPU is detected.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
