import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import time
from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, AutoConfig,
        GlueDataTrainingArguments,
        )
from datasets import load_dataset, load_metric
import numpy as np
from ray import tune
import random
def set_random_seed(seed = 42):
    print('seed for random sampling : {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def experiment_setup():
    return './test' + str(time.time())
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("kssteven/ibert-roberta-base")
    return tokenizer

def get_dataset():
    dataset = load_dataset('glue', 'cola')
    return dataset

def get_metric():
    metric_acc = load_metric('accuracy')
    metric_metthew = load_metric('glue', 'cola')
    
    return metric_acc, metric_metthew

def encode(examples):
    outputs = tokenizer(examples['sentence'], truncation = True, padding = True)
    return outputs

def get_model(config):
    configuration = AutoConfig.from_pretrained("kssteven/ibert-roberta-base")
    if config is not None:
        print(config)
        configuration.update({'hidden_dropout_prob' : config['hidden_dropout_prob'], 'attention_dropout_prob' : config['attention_dropout_prob']})
    model = AutoModelForSequenceClassification.from_pretrained("kssteven/ibert-roberta-base", config = configuration)
    return model 

def get_param_space():
    space = {
            "learning_rate" : tune.grid_search([1e-5, 2e-5, 3e-5]),
            "hidden_dropout_prob" : tune.grid_search([0.1]),
            "attention_dropout_prob" : tune.grid_search([0.1])
            }
    return space

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis = -1)
    result = metric_metthew.compute(predictions = predictions, references = labels)
    accuracy = metric_acc.compute(predictions = predictions, references = labels)
    result.update(accuracy)
    return result

def objective_metric(metrics):
    return metrics["eval_matthews_correlation"]

if __name__ == "__main__":
    set_random_seed()
    tokenizer = get_tokenizer()
    metric_acc, metric_metthew = get_metric()
    space = get_param_space()
    training_args = TrainingArguments(
            "./test/" + str(time.time()), per_device_train_batch_size = 32, per_device_eval_batch_size = 32, evaluation_strategy = "steps", eval_steps = 100,
            do_train = True, do_eval = True, weight_decay = 0.1, adam_beta1 = 0.9, adam_beta2 = 0.98, adam_epsilon = 1e-06, lr_scheduler_type = "linear",
            warmup_ratio = 0.06, logging_strategy = "steps", logging_dir = "./test/" + str(time.time()) + "/log", logging_steps = 20, save_strategy = "steps", save_steps = 100,save_total_limit = 2, load_best_model_at_end = True,
            metric_for_best_model = "matthews_correlation", greater_is_better = True, num_train_epochs = 10,
            )
    
    dataset = get_dataset()
    encoded_dataset = dataset.map(encode, batched = True)

    trainer = Trainer(
            args = training_args,
            tokenizer = tokenizer,
            train_dataset = encoded_dataset['train'],
            eval_dataset = encoded_dataset['validation'],
            model_init = get_model,
            compute_metrics = compute_metrics
            )

    trainer.hyperparameter_search(
            hp_space = lambda _ : space,
            compute_objective = objective_metric,
            n_trials = 3,
            direction = "maximize",
            backend = "ray",
            )
