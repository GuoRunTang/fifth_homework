
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time


model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


summarizer = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)


from datasets import load_dataset
dataset = load_dataset("cnn_dailymail")

articles = dataset["test"]["article"]
summaries = dataset["test"]["highlights"]


from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
bart_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
bart_classifier = pipeline("zero-shot-classification", model=bart_model, tokenizer=bart_tokenizer)
def compute_bartscore(generated_summary, reference_summary):
  result = bart_classifier(generated_summary, [reference_summary])
  score = result["scores"][0]
  return score


total_score = 0
total_time = 0
for article, summary in zip(articles[:10], summaries[:10]): 
  start_time = time.time()
  result = summarizer(article, [summary]) 
  end_time = time.time()
  generated_summary = result["labels"][0]
  score = compute_bartscore(generated_summary, summary)
  total_score += score
  total_time += end_time - start_time
  print(f"Article: {article}")
  print(f"Summary: {summary}")
  print(f"Generated summary: {generated_summary}")
  print(f"BARTScore: {score}")
  print()

average_score = total_score / len(articles[:10])
average_speed = len(articles[:10]) / total_time
print(f"Average BARTScore: {average_score}")
print(f"Average speed: {average_speed} articles/second")