import numpy as np
import pandas as pd

class ModelExplainer:
    def __init__(self, model, vectorizer, label_encoder):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.feature_names = np.array(vectorizer.get_feature_names_out())
        
    def explain_prediction(self, text, preprocessor, top_n=10):
        """Explain why a single prediction was made using TF-IDF weights and Model Coefficients."""
        cleaned_text = preprocessor.full_pipeline(text)
        tfidf_vector = self.vectorizer.transform([cleaned_text])
        
        prediction_idx = self.model.predict(tfidf_vector)[0]
        predicted_class = self.label_encoder.inverse_transform([prediction_idx])[0]
        
        # Get probability if available
        prob = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(tfidf_vector)[0]
            prob = probs[prediction_idx]
            
        # Extract features present in the document
        doc_features_indices = tfidf_vector.nonzero()[1]
        doc_features = self.feature_names[doc_features_indices]
        doc_tfidf_scores = tfidf_vector.toarray()[0, doc_features_indices]
        
        # Determine feature importance based on the model type
        feature_importance = []
        
        if hasattr(self.model, 'coef_'):
            # Linear models (Logistic Regression, SVM, Naive Bayes)
            coefs = self.model.coef_
            if coefs.ndim > 1:
                # Multiclass: use the coefficients for the predicted class
                class_coefs = coefs[prediction_idx, doc_features_indices]
            else:
                # Binary (fallback)
                class_coefs = coefs[0, doc_features_indices]
                if prediction_idx == 0:
                    class_coefs = -class_coefs
                    
            for word, tfidf_score, coef in zip(doc_features, doc_tfidf_scores, class_coefs):
                importance = tfidf_score * coef
                feature_importance.append((word, tfidf_score, coef, importance))
                
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based models (Random Forest)
            importances = self.model.feature_importances_[doc_features_indices]
            for word, tfidf_score, imp in zip(doc_features, doc_tfidf_scores, importances):
                importance = tfidf_score * imp
                feature_importance.append((word, tfidf_score, imp, importance))
        else:
             # Fallback to just TF-IDF scores if model doesn't support coefficients easily
             for word, tfidf_score in zip(doc_features, doc_tfidf_scores):
                feature_importance.append((word, tfidf_score, 0, tfidf_score))

        # Sort by final importance
        feature_importance.sort(key=lambda x: x[3], reverse=True)
        top_features = feature_importance[:top_n]
        
        return {
            'Predicted_Class': predicted_class,
            'Confidence': prob,
            'Top_Features': top_features,
            'Cleaned_Text': cleaned_text
        }
