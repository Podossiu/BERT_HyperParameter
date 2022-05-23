from datasets import load_metric

metric = load_metric("glue", "cola")
metric_acc = load_metric("accuracy")
a = metric.compute(predictions = [0,0,1,1], references = [0,0,1,1])
b = metric_acc.compute(predictions = [0,0,1,1], references = [0,0,1,1])
a.update(b)
print(a)
