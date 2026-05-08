from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Evaluator:
    def __init__(self, y_true, y_pred, classes=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes

    def get_metrics(self):
        return {
            'Accuracy': accuracy_score(self.y_true, self.y_pred),
            'Precision': precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        }

    def get_classification_report(self):
        return classification_report(self.y_true, self.y_pred, target_names=self.classes, zero_division=0)

    def plot_confusion_matrix(self, save_path=None):
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            return plt.gcf()
