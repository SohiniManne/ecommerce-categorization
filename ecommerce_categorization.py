# E-commerce Product Categorization ML Project
# This project classifies products into categories based on their titles and descriptions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("Warning: Some NLTK data couldn't be downloaded. Using alternative preprocessing.")

class EcommerceProductCategorizer:
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.model = None
        
        # Initialize NLTK components with fallbacks
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Warning: NLTK components not available. Using basic preprocessing.")
            self.lemmatizer = None
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
    def create_sample_data(self, n_samples=1000):
        """Create sample e-commerce product data"""
        np.random.seed(42)
        
        # Define categories and sample products
        categories = {
            'Electronics': [
                'smartphone mobile phone', 'laptop computer notebook', 'tablet ipad android',
                'headphones earbuds wireless', 'smartwatch fitness tracker', 'camera digital photography',
                'speaker bluetooth portable', 'monitor display screen', 'keyboard mouse wireless',
                'gaming console playstation xbox', 'television tv smart', 'router wifi network'
            ],
            'Clothing': [
                'shirt cotton casual', 'jeans denim pants', 'dress summer casual',
                'jacket winter coat', 'shoes sneakers running', 'boots leather hiking',
                'sweater wool knit', 't-shirt cotton graphic', 'shorts summer casual',
                'hoodie sweatshirt comfortable', 'skirt mini maxi', 'blazer formal office'
            ],
            'Home & Garden': [
                'sofa couch furniture', 'table dining wooden', 'chair office ergonomic',
                'lamp lighting decorative', 'curtains blinds window', 'rug carpet floor',
                'plant indoor succulent', 'tool hammer screwdriver', 'paint wall interior',
                'garden hose watering', 'furniture bedroom storage', 'kitchen utensils cooking'
            ],
            'Sports & Outdoors': [
                'basketball sports equipment', 'yoga mat exercise fitness', 'bicycle mountain bike',
                'tennis racket sports', 'camping tent outdoor', 'hiking boots trail',
                'swimming goggles pool', 'football soccer ball', 'golf clubs equipment',
                'fishing rod tackle', 'skateboard longboard', 'dumbbells weights fitness'
            ],
            'Books': [
                'novel fiction mystery', 'cookbook recipes cooking', 'textbook educational learning',
                'biography memoir autobiography', 'science fiction fantasy', 'romance love story',
                'history world war', 'self-help motivation', 'children kids picture book',
                'travel guide vacation', 'art design coffee table', 'poetry poems literature'
            ],
            'Beauty & Health': [
                'shampoo hair care', 'moisturizer skin care', 'makeup cosmetics foundation',
                'perfume fragrance cologne', 'vitamins supplements health', 'toothbrush dental care',
                'sunscreen protection spf', 'nail polish manicure', 'face mask skincare',
                'protein powder fitness', 'essential oils aromatherapy', 'razor shaving grooming'
            ]
        }
        
        # Generate sample data
        data = []
        for category, products in categories.items():
            for _ in range(n_samples // len(categories)):
                # Random product from category
                base_product = np.random.choice(products)
                
                # Add some variations
                adjectives = ['premium', 'professional', 'affordable', 'high-quality', 'durable', 
                            'lightweight', 'comfortable', 'stylish', 'modern', 'classic']
                colors = ['black', 'white', 'blue', 'red', 'green', 'silver', 'gold', 'pink']
                sizes = ['small', 'medium', 'large', 'xl', 'compact', 'mini', 'jumbo']
                
                # Create product title
                title_parts = [base_product]
                if np.random.random() > 0.5:
                    title_parts.append(np.random.choice(adjectives))
                if np.random.random() > 0.7:
                    title_parts.append(np.random.choice(colors))
                if np.random.random() > 0.8:
                    title_parts.append(np.random.choice(sizes))
                
                title = ' '.join(title_parts)
                
                # Create description
                description = f"High-quality {base_product} perfect for daily use. " \
                            f"Features excellent build quality and comes with warranty. " \
                            f"Ideal for {category.lower()} enthusiasts."
                
                data.append({
                    'title': title,
                    'description': description,
                    'category': category
                })
        
        return pd.DataFrame(data)
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Simple tokenization (fallback if NLTK fails)
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split if NLTK fails
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        try:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        except:
            # Fallback without lemmatization if NLTK fails
            basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
            tokens = [token for token in tokens if token not in basic_stopwords and len(token) > 2]
        
        return ' '.join(tokens)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Combine title and description
        df['combined_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        
        # Preprocess text
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        return df
    
    def train_model(self, df, test_size=0.2, random_state=42):
        """Train the categorization model"""
        # Prepare data
        df = self.prepare_data(df)
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['category'])
        X_text = df['processed_text']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train multiple models and select the best one
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(kernel='linear', random_state=42)
        }
        
        best_score = 0
        best_model_name = ""
        
        print("Model Performance Comparison:")
        print("-" * 50)
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
            mean_score = cv_scores.mean()
            
            print(f"{name}: {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                self.model = model
        
        print(f"\nBest Model: {best_model_name}")
        print("-" * 50)
        
        # Train the best model
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_vec)
        
        print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Store test data for visualization
        self.X_test = X_test_vec
        self.y_test = y_test
        self.y_pred = y_pred
        
        return self.model
    
    def predict(self, title, description=""):
        """Predict category for a new product"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Combine and preprocess text
        combined_text = f"{title} {description}"
        processed_text = self.preprocess_text(combined_text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        # Get category name
        category = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities for all categories
        categories = self.label_encoder.classes_
        prob_dict = dict(zip(categories, probability))
        
        return category, prob_dict
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        if not hasattr(self, 'y_test'):
            print("No test data available. Train the model first.")
            return
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance for tree-based models"""
        if not hasattr(self.model, 'feature_importances_'):
            print("Feature importance not available for this model type.")
            return
        
        # Get feature names and importance
        feature_names = self.vectorizer.get_feature_names_out()
        importance = self.model.feature_importances_
        
        # Get top features
        top_indices = np.argsort(importance)[-top_n:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the project"""
    print("E-commerce Product Categorization ML Project")
    print("=" * 50)
    
    # Initialize categorizer
    categorizer = EcommerceProductCategorizer()
    
    # Create sample data
    print("Creating sample data...")
    df = categorizer.create_sample_data(n_samples=1200)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Categories: {df['category'].unique()}")
    print(f"Category distribution:\n{df['category'].value_counts()}")
    
    # Train model
    print("\nTraining models...")
    model = categorizer.train_model(df)
    
    # Test predictions
    print("\nTesting predictions:")
    print("-" * 30)
    
    test_products = [
        ("iPhone 13 Pro Max", "Latest smartphone with advanced camera"),
        ("Nike Running Shoes", "Comfortable athletic footwear for running"),
        ("Wooden Dining Table", "Beautiful oak dining table for 6 people"),
        ("Yoga Mat Premium", "Non-slip exercise mat for yoga and fitness"),
        ("Python Programming Book", "Learn programming with practical examples"),
        ("Face Moisturizer", "Hydrating cream for all skin types")
    ]
    
    for title, description in test_products:
        category, probabilities = categorizer.predict(title, description)
        print(f"Product: {title}")
        print(f"Predicted Category: {category}")
        print(f"Confidence: {max(probabilities.values()):.3f}")
        print("-" * 30)
    
    # Visualizations
    print("\nGenerating visualizations...")
    categorizer.plot_confusion_matrix()
    categorizer.plot_feature_importance()
    
    # Interactive prediction function
    def interactive_prediction():
        print("\nInteractive Product Categorization")
        print("Enter 'quit' to exit")
        
        while True:
            title = input("\nEnter product title: ")
            if title.lower() == 'quit':
                break
            
            description = input("Enter product description (optional): ")
            
            try:
                category, probabilities = categorizer.predict(title, description)
                print(f"\nPredicted Category: {category}")
                print("All probabilities:")
                for cat, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {cat}: {prob:.3f}")
            except Exception as e:
                print(f"Error: {e}")
    
    # Uncomment the line below to enable interactive mode
    # interactive_prediction()

if __name__ == "__main__":
    main()