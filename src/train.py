import os
import time
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.data_loader import load_resume_data
from src.preprocessing import NLPPreprocessor
from src.feature_engineering import FeatureEngineer
from src.evaluate import Evaluator

def train_models():
    print("Loading data...")
    df = load_resume_data()
    
    print("Preprocessing text...")
    preprocessor = NLPPreprocessor()
    df['Cleaned_Resume'] = df['Resume'].apply(lambda x: preprocessor.full_pipeline(x))
    
    print("Encoding labels...")
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Category'])
    
    # Save label encoder
    os.makedirs('models', exist_ok=True)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Resume'], df['Label'], test_size=0.2, random_state=42)
    
    print("Feature Engineering (TF-IDF)...")
    fe = FeatureEngineer()
    X_train_tfidf = fe.fit_transform_tfidf(X_train)
    X_test_tfidf = fe.transform_tfidf(X_test)
    fe.save_models('models/')
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='linear', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    print("Training models...")
    for name, model in models.items():
        start_train = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - start_train
        
        start_infer = time.time()
        y_pred = model.predict(X_test_tfidf)
        infer_time = time.time() - start_infer
        
        # Save each model
        with open(f'models/{name.replace(" ", "_").lower()}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        trained_models[name] = model
        
        evaluator = Evaluator(y_test, y_pred, le.classes_)
        metrics = evaluator.get_metrics()
        metrics['Model'] = name
        metrics['Train Time (s)'] = round(train_time, 4)
        metrics['Inference Time (s)'] = round(infer_time, 4)
        results.append(metrics)
        print(f"{name} trained. Accuracy: {metrics['Accuracy']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/reports/model_comparison.csv', index=False)
    print("Training complete. Results saved.")

if __name__ == "__main__":
    train_models()
