import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_and_preprocess_data

data_dir = '../data/Chessman'
val_data = load_and_preprocess_data(data_dir)

model = load_model('best_model.h5')

val_labels = val_data.classes
val_preds = model.predict(val_data)
val_preds_classes = np.argmax(val_preds, axis=1)

print(classification_report(val_labels, val_preds_classes))

conf_matrix = confusion_matrix(val_labels, val_preds_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()