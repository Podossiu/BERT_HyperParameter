import os

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import (
        download_data,
        build_compute_metrics_fn,
)

from ray.tune.scheduler import PopulationBasedTraining
from transformers import (
        glue_tasks_num_labels,
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        GlueDataset,
        GlueDataTrainingArguments,
        TrainingArguments,
)

def tune_transformer(num_samples = 8, gpus_per_trial = 0, smoke_test = False):
    data_dir_name = "./data" if not smoke_test else "./test_data"
    data_dir = os.path.abspath(os.path.join(os.getcwd(), data_dir_name))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, 0o755)

    model_name = (
            "bert-base-uncased" if not smoke_test else "RoBERTa-based")

    task_name = "cola"

    task_data_dir = os.path.join(data_dir, task_name.upper())

    num_labels = glue_tasks_num_labels[task_name]

    config = AutoConfig.from_pretrained(
            model_name, num_labels = num_labels, finetuning_task = task_name
    )
    
    print("Downloading and caching Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Downloading and caching pre-trained model")
    AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config = config
    )

    def get_model():
        return AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config = config
        )

    download_data(task_name, data_dir)

    data_args = GlueDataTrainingArguments(task_name = task_name, data_dir = task_data_dir)
    
    train_dataset = GlueDataset(
            data_args, tokenizer = tokenizer, mode = 'train', cache_dir = task_data_dir
    )
    eval_dataset = GlueDataset(
            data_args, tokenizer = tokenizer, mode = 'dev', cache_dir = task_data_dir
    )

    training_args = TrainingArguments(
            output_dir = ".",
            learning_rate = 1e-5,
            do_train = True,
            do_eval = True,
            no_cuda = (gpus_per_trail <= 0),
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            load_best_model_at_end = True,
            num_train_epochs = 6,
            max_steps = -1,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size = 16,
            warmup_steps = 0,
            weight_decay = 0.1,
            logging_dir = "./logs",
            skip_memory_metrics = True,
            report_to = "none",
    )
    
    trainer = Trainer(
            model_init = get_model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = build_compute_metrics_fn(task_name),
    )

    tune_config = {
            "per_device_train_batch_size" : 32,
            "per_device_eval_batch_size" : 32,
            "num_train_epochs" : tune.choice([2,3,4,5]),
            "max_steps" : 1 if smoke_test else -1,
    }

    scheduler = PopulationBasedTraining

