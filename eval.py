from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from datasets import load_dataset, load_metric
import numpy as np
from transformers import Trainer,TrainingArguments

def args_parse():
    parser = argparse.ArgumentParser(
            description = "Evalutation for NLP")
    parser.add_argument('--do_eval', action = "store_true")
    parser.add_argument('--ckpt-dir', type = str)
    parser.add_argument('--arch', type = str)
    parser.add_argument('--max-sentences', type = int, default = 16)
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--quant-mode', type = str, default = "none")

    args = parser.parse_args()
    return args

def get_model(args):
    if args.arch == "roberta_base":
        model = AutoModelForSequenceClassification.from_pretrained(args.ckpt_dir, local_files_only = True)

    if args.quant_mode == "none":
        model.quant_mode = False
    else:
        model.quant_mode = True
    return model

def get_metric(args):
    if args.dataset == "CoLA":
        metric = load_metric('glue', 'cola')
    return metric
def get_tokenizer(args):
    if args.arch == "roberta_base":
        tokenizer = AutoTokenizer.from_pretrained('kssteven/ibert-roberta-base')
    return tokenizer

def get_dataset(args):
    if args.dataset == "CoLA":
        dataset = load_dataset("glue", "cola")
    return dataset

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding = "max_length", truncation = True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions = predictions, references = labels)

if __name__ == "__main__":
    args = args_parse()
    print(args)
    tokenizer = get_tokenizer(args)
    dataset = get_dataset(args)
    tokenized_data = dataset.map(tokenize_function, batched = True)

    model = get_model(args)
    metric = get_metric(args)
    training_args = TrainingArguments('./', save_strategy = 'no')
    trainer = Trainer(model, args = training_args,eval_dataset = tokenized_data['validation'], compute_metrics = compute_metrics)
    trainer.evaluate(tokenized_data['validation'])
