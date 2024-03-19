import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

scored_data = pd.read_csv('/home/katie/Desktop/BirdNET-Analyzer/formatted_data_for_20170422_180000.csv')
ml_output = pd.read_csv('/home/katie/Desktop/BirdNET-Analyzer/adjusted_ml_output.csv')


y_true = scored_data['Label'].map({'yes': 1, 'no': 0}).values
y_pred = ml_output['Label'].map({'yes': 1, 'no': 0}).values

cm = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

