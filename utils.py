# utils.py
import csv
import json
import numpy as np

def load_csv(path):
    texts, labels = [], []
    with open(path, "r", encoding="latin1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["OriginalTweet"])
            labels.append(row["Sentiment"])
    return texts, labels

def save_json(obj, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    import json
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def compute_precision_recall_f1(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    results = {}
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        results[c] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return results, macro_f1
