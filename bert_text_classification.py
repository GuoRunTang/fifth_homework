import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)


from datasets import load_dataset
dataset = load_dataset("glue", "cola")

texts = dataset["test"]["sentence"]
labels = dataset["test"]["label"]


def convert_label(label):
  if label == 0:
    return ["unacceptable", "acceptable"]
  else:
    return ["acceptable", "unacceptable"]


correct = 0
total = 0
start_time = time.time()
for text, label in zip(texts, labels):
  result = classifier(text, convert_label(label))
  prediction = result["labels"][0]
  if prediction == convert_label(label)[0]:
    correct += 1
  total += 1

end_time = time.time()
accuracy = correct / total
speed = total / (end_time - start_time)
print(f"Accuracy: {accuracy}")
print(f"Speed: {speed} texts/second")