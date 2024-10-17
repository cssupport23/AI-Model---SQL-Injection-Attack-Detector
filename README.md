# SQL Injection Detection Using Hugging Face Transformers

This project fine-tunes the `google/mobilebert-uncased` model from Hugging Face's Transformers library to detect SQL injection attacks. The fine-tuning process is customizable, supporting Nvidia's APEX for mixed precision training and leveraging optimizers like Adafactor or AdamW. The model is trained on a custom dataset and uses techniques such as dynamic padding and learning rate scheduling.

This fine-tuned model, for detecting SQL injection attacks is now publicly available on [Hugging Face](https://huggingface.co/cssupport/mobilebert-sql-injection-detect). This model is optimized for identifying malicious SQL queries and can be used for a variety of security applications, such as web application firewalls, intrusion detection systems, and API security. Leveraging transfer learning from the `google/mobilebert-uncased` model, it has been further fine-tuned using a custom SQL injection dataset. You can easily integrate the model into your security pipeline for real-time detection.

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


## Execution Sequence

1. **Initialization of the Pipeline:**
   - The process begins by initializing an instance of the `SQLInjectionPipeline` class. The constructor (`__init__`) method loads the model and any pre-configured settings from the `config` dictionary.
   
   ```python
   pipeline = SQLInjectionPipeline(config)
   ```

2. **Data Loading:**
   - The `setup_and_train()` method starts by loading the dataset from a CSV file specified in the config (`data_file`). The `load_data()` method reads the CSV, extracts SQL queries and their labels, and then tokenizes them using the Hugging Face tokenizer for MobileBERT.
   
   ```python
   train_data, eval_data = pipeline.load_data()
   ```

3. **Data Preprocessing and Tokenization:**
   - Tokenization converts each SQL query into input IDs and attention masks. It uses dynamic padding to ensure that batches are padded to the maximum sequence length in the batch, rather than the entire dataset, improving efficiency.

4. **Model Initialization:**
   - If there’s a fine-tuned model specified in the configuration (`ft_model`), it loads the weights from the saved model. Otherwise, it loads the pre-trained MobileBERT model from Hugging Face (`google/mobilebert-uncased`) for sequence classification.
   
   ```python
   model = pipeline.initialize_model()
   ```

5. **Optimizer and Scheduler Setup:**
   - The optimizer (either `Adafactor` or `AdamW`) is initialized. The choice depends on the config. 
     - **Adafactor** is often used when handling large models because it requires less memory.
     - **AdamW** is a standard optimizer that applies weight decay to prevent overfitting.

   The learning rate scheduler is initialized as well. The scheduler is responsible for adjusting the learning rate dynamically over the training period. A **linear scheduler** with a warmup phase is used here, which gradually increases the learning rate for a number of steps (defined by `warmup_steps`) and then decreases it linearly.

   ```python
   optimizer = self.get_optimizer()
   scheduler = self.get_scheduler(optimizer, warmup_steps=config['warmup_steps'])
   ```

6. **Training Loop:**
   - The `train()` method handles the core training logic. It runs over a number of epochs (`EPOCHS` defined in the config).
   - For each batch in the dataset:
     1. **Forward Pass**: The input data is fed into the model to compute predictions.
     2. **Loss Calculation**: The model's predictions are compared to the actual labels using cross-entropy loss.
     3. **Backward Pass**: The gradients are computed using backpropagation.
     4. **Optimizer Step**: The optimizer updates the model's parameters using the computed gradients.
     5. **Scheduler Step**: The learning rate is updated according to the current step in training.
   
   If mixed precision training is enabled (`use_amp`), the model performs forward and backward passes in mixed precision, reducing memory consumption and speeding up the training process.

   ```python
   results = pipeline.train(train_data, eval_data)
   ```

7. **Evaluation and Metrics Calculation:**
   - After each epoch, the model is evaluated on the validation dataset using precision, recall, F1-score, and ROC-AUC. These metrics are calculated using the `evaluate()` method. Results are printed for each epoch to monitor progress.
   
   ```python
   eval_metrics = pipeline.evaluate(eval_data)
   ```

8. **Model Saving:**
   - After each epoch, the model's state is saved to the location specified in the configuration. Each model checkpoint includes the epoch number and version for easy tracking.

   ```python
   self.save_model(epoch)
   ```

### Optimizer's Role

The **optimizer** is responsible for updating the model's parameters to minimize the loss function during training.

- **AdamW**: This optimizer adds weight decay (regularization) to reduce overfitting. It is effective for general-purpose training and balances between the momentum of gradient updates and adjusting learning rates on the fly.
  
- **Adafactor**: Adafactor is a memory-efficient optimizer, particularly useful for models like MobileBERT. It adapts learning rates across dimensions, requiring less memory and improving scalability when working with large models or datasets.

In both cases, the optimizers update the model’s weights by using gradients (computed during backpropagation) to take steps toward minimizing the loss function.

### Scheduler's Role

The **scheduler** adjusts the learning rate during training, ensuring smoother training and preventing overshooting of the optimal parameters.

- **Warmup and Linear Decay Scheduler**: 
   - **Warmup**: During the initial training phase, the learning rate starts low and gradually increases over a number of steps (defined by `warmup_steps`). This helps stabilize training during the early epochs when gradients might be large and noisy.
   - **Linear Decay**: After the warmup phase, the learning rate decreases linearly throughout the remaining steps of training. This ensures that the model makes smaller adjustments as it converges, improving the fine-tuning process and achieving a more stable end result.

### Detailed Methods

#### `load_data()`
- Reads the input data from a CSV file, splits it into training and validation sets, and tokenizes the SQL queries using the Hugging Face tokenizer.

#### `initialize_model()`
- Initializes the `google/mobilebert-uncased` model from Hugging Face or loads a fine-tuned model if specified.

#### `get_optimizer()`
- Instantiates the optimizer based on the config. It supports Adafactor and AdamW, and sets parameters such as learning rate and weight decay.

#### `get_scheduler(optimizer, warmup_steps)`
- Creates a learning rate scheduler that controls how the learning rate evolves over time. This is usually combined with the optimizer to improve training stability.

#### `train()`
- Contains the core training loop, including forward passes, loss computation, backward propagation, optimizer updates, and scheduler steps.

#### `evaluate()`
- Evaluates the model on the validation set using metrics like precision, recall, F1-score, and ROC-AUC.

#### `save_model(epoch)`
- Saves the model state at the end of each epoch for checkpointing.

### Conclusion

The execution flow and methods ensure that the model is trained efficiently, with optimizers and schedulers playing a crucial role in controlling how the model learns. The optimizer updates the model's weights, while the scheduler fine-tunes the learning rate dynamically to stabilize training. This combination helps achieve robust fine-tuning of the MobileBERT model for SQL injection detection.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
