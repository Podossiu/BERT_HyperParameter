from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric
import argparse
from datasets import load_dataset

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding 
parser = argparse.ArgumentParser(description= "Quantization process")
parser.add_argument('-q', '--quant', type = str)


args = parser.parse_args()
metric = load_metric("accuracy")
dataset = load_dataset("glue", "cola")

tokenizer = AutoTokenizer.from_pretrained("kssteven/ibert-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("kssteven/ibert-roberta-base", num_labels = 2)
model.config.quant_mode = args.quant

training_args = TrainingArguments(output_dir = './outputs', evaluation_strategy = "epoch")

trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = small_train_dataset,
        eval_dataset = small_eval_dataset,
        compute_metrics = metric
        )
