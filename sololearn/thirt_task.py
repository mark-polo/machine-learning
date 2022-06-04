tp, fp, fn, tn = [int(x) for x in input().split()]

Accuracy = (tp + tn) / (tp + fp + fn + tn)

Precision = tp / (tp + fp)

Recall = tp / (tp + fn)

F1_score = 2 * Recall * Precision / (Precision + Recall)

print(round(Accuracy, 4))
print(round(Precision, 4))
print(round(Recall, 4))
print(round(F1_score, 4))