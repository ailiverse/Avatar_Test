import requests
import pandas as pd

qna = pd.read_csv("QA.csv")

tp = 0
fp = 0
fn = 0
tn = 0

for i in range(len(qna)):
    row = qna.iloc[i]
    question = row["Question"]
    answer = row["Answer"]

    results = requests.post("http://34.126.164.20:8085/", json={"question": question, "Model": "AI"}).json()["results"]
    if results.strip() == answer.strip():
        tp += 1
    else:
        fp += 1

precision = tp/(tp + fp)
recall = tp/(tp + fn)
F1 = 2/(1/precision + 1/recall)

print(f"f1 score {F1}, precision: {precision}, recall: {recall}")
