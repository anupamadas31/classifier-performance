from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(Y_test, prediction, pos_label=1)
recall_score(Y_test, prediction, pos_label=1)
f1_score(Y_test, prediction, pos_label=1)

from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)
print(report)

pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.2, 0.1)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
for pred, y in zip(pos_prob, Y_test):
  for i, threshold in enumerate(thresholds):
    if pred >= threshold:
# if truth and prediction are both 1
      if y == 1:
       true_pos[i] += 1
# if truth is 0 while prediction is 1
      else:
        false_pos[i] += 1
    else:
     break

true_pos_rate = [tp /num_of_positive_testing_samples for tp in true_pos]
false_pos_rate = [fp /num_of_negative_testing_samples  for fp in false_pos]

#plotting ROC curve
import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color='darkorange',
lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#computing AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, pos_prob)