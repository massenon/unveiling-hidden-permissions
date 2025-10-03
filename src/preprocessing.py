# src/preprocessing.py
import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

def preprocess_review(text):
    """Cleans and lemmatizes a user review."""
    doc = nlp(text.lower()) # Lowercase
    
    # Lemmatize and remove stop words and punctuation
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]
    return tokens

if __name__ == '__main__':
    review = "This app is great! But it constantly tracks my location??"
    processed_tokens = preprocess_review(review)
    print(f"Original: {review}")
    print(f"Processed: {processed_tokens}")
    # Expected output: ['app', 'great', 'constantly', 'track', 'location']