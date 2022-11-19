# Accuracy score:
from sklearn.metrics import accuracy_score
print('Accuracy: ', accuracy_score(y_test, predictions))

# Classification report
# gives precision, recall and f1-score. Also support (number of cases and avgs, aso weighted to account for bias in the number of cases predicted)
from sklearn. metrics import classification_report
print(classification_report(y_test, predictions))

# Exact Precision and Recall
from sklearn.metrics import precision_score, recall_score

print("Overall Precision:",precision_score(y_test, predictions))
print("Overall Recall:",recall_score(y_test, predictions))

# Confusion matrix
from sklearn.metrics import confusion_matrix
# Print the confusion matrix
cm = confusion_matrix(y_test, predictions)
print (cm)


# Calculation of Accuracy, Precision and Recall
# Accuracy = (TP+TN)/(TP+TN+FP+FN) How many predictions were correct
# Recall = TP/(TP+FN) Of all, how many positives were identified
# Precision = TP/(TP+FP) How many of the positive predicted are actually positive

# Give Propabilities
y_scores = model.predict_proba(X_test)
print(y_scores)

# ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# AUC is the are under the ROC curve, 1 is perfect, 0.5 would be guessing

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))

# Open Questions:
# How to influence the threshold (base=0.5) to give more emphasize on precision or recall
