import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class StyleSenseProcessor:
    """Custom processor for fashion review data"""
    
    def __init__(self):
        self.fashion_terms = {
            'fitin': 'fitting', 'comfertable': 'comfortable', 'stylysh': 'stylish',
            'durablee': 'durable', 'qualite': 'quality', 'pricyy': 'pricey'
        }
    
    def fashion_clean(self, text):
        """Unique fashion-specific cleaning pipeline"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase and normalize spaces
        text = str(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep fashion-relevant ones
        text = re.sub(r'[^a-zA-Z0-9\s\.\-]', ' ', text)
        
        # Custom fashion typo corrections
        for wrong, right in self.fashion_terms.items():
            text = re.sub(rf'\b{wrong}\b', right, text)
        
        # Remove short words (<3 chars) and extra spaces
        words = [w for w in text.split() if len(w) > 2]
        return ' '.join(words)

# Load and prepare data
print("🚀 Loading StyleSense fashion review data...")
df = pd.read_csv('fashion_reviews.csv')  # Assumes 'review_text', 'recommended' columns

# Apply unique preprocessing
processor = StyleSenseProcessor()
df['clean_review'] = df['review_text'].apply(processor.fashion_clean)

print(f"✅ Processed {len(df)} reviews")
print("Sample processed text:", df['clean_review'].iloc[0])

# Train-test split (75/25 for better validation)
X = df['clean_review']
y = df['recommended']  # 0=no, 1=yes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123, stratify=y
)

## 🛠️ CREATE UNIQUE MODEL PIPELINE
print("\n🔨 Building StyleSense ML Pipeline...")

stylesense_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        max_features=4000,
        min_df=3,
        ngram_range=(1, 3),  # Capture fashion phrases like "great fit"
        stop_words=None  # Custom handling
    )),
    ('classifier', RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=123,
        n_jobs=-1
    ))
])

# Fit pipeline
stylesense_pipeline.fit(X_train, y_train)

## ⚙️ HYPERPARAMETER TUNING (RandomizedSearchCV - unique approach)
print("\n🎛️ Fine-tuning hyperparameters...")
param_distributions = {
    'vectorizer__max_features': [3500, 4500, 5500],
    'vectorizer__ngram_range': [(1,2), (1,3)],
    'classifier__n_estimators': [120, 180, 220],
    'classifier__max_depth': [10, 14, 18]
}

tuner = RandomizedSearchCV(
    stylesense_pipeline,
    param_distributions,
    n_iter=12,  # Fewer iterations = faster
    cv=4,
    scoring='accuracy',
    random_state=123,
    n_jobs=-1
)

tuner.fit(X_train, y_train)
best_model = tuner.best_estimator_

print(f"🎯 Best Parameters: {tuner.best_params_}")
print(f"📊 Best CV Score: {tuner.best_score_:.3f}")

## 📈 EVALUATE FINAL MODEL
print("\n📊 Evaluating on test data...")
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

print("\n✨ StyleSense Pipeline Complete!")
print("Ready for GitHub deployment 🚀")
