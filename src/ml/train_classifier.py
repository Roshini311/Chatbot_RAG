import os
import pickle
import numpy as np
from config.settings import settings
from src.ml.dataset_prep import get_training_data, CATEGORIES

def train_and_save_classifier():
    """
    Trains a document domain classifier.
    Tries TensorFlow first; falls back to scikit-learn TF-IDF classifier if TF is unavailable/incompatible.
    """
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    texts, labels = get_training_data()

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        # Expand training data artificially for neural network stability
        expanded_texts = texts * 10
        expanded_labels = labels * 10

        # Vectorizer
        vocab_size = 1000
        max_len = 100

        vectorize_layer = layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=max_len
        )
        vectorize_layer.adapt(expanded_texts)

        model = models.Sequential([
            vectorize_layer,
            layers.Embedding(vocab_size, 32, mask_zero=True),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(CATEGORIES), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            np.array(expanded_texts), 
            np.array(expanded_labels), 
            epochs=15, 
            batch_size=8, 
            verbose=0
        )

        # Save model
        tf_model_path = settings.MODEL_PATH
        if not tf_model_path.endswith(".h5") and not tf_model_path.endswith(".keras"):
            tf_model_path += ".h5"
        model.save(tf_model_path)

        with open(settings.TOKENIZER_PATH, "wb") as f:
            pickle.dump({"backend": "tf", "categories": CATEGORIES}, f)

        print(f"Successfully trained & saved TensorFlow Document Classifier to {tf_model_path}")
        return True

    except Exception as e:
        print(f"TensorFlow training notice ({e}). Using Scikit-Learn TF-IDF Fallback Engine...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB

        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts * 10)
        y = labels * 10

        clf = MultinomialNB()
        clf.fit(X, y)

        model_path = settings.MODEL_PATH.replace(".h5", ".pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        with open(settings.TOKENIZER_PATH, "wb") as f:
            pickle.dump({"backend": "sklearn", "vectorizer": vectorizer, "categories": CATEGORIES}, f)

        print(f"Successfully trained & saved Fallback Document Classifier to {model_path}")
        return True

if __name__ == "__main__":
    train_and_save_classifier()
