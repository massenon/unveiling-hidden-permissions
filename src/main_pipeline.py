# src/main_pipeline.py
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from .preprocessing import preprocess_review # Use relative import

# Assume model is trained and saved in './models/fine_tuned_roberta'
# For demonstration, we'll use a public sentiment analysis model
classifier = pipeline(
    'sentiment-analysis', 
    model="siebert/sentiment-roberta-large-english"
)

def jaccard_similarity(set1, set2):
    """Calculates Jaccard Similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def analyze_review_and_permissions(review_text, declared_permissions):
    """
    Main pipeline to analyze a review against declared permissions.
    """
    # 1. Classify the review to get risk and confidence score
    # In a real system, this would be your fine-tuned model
    classification_result = classifier(review_text)[0]
    nlp_score = classification_result['score']
    # The label would be one of your risk categories, e.g., 'Unauthorized Tracking'
    risk_category = classification_result['label'] 

    print(f"Review classified as '{risk_category}' with score: {nlp_score:.2f}")

    # 2. Preprocess review to get keywords
    review_keywords = set(preprocess_review(review_text))
    
    # 3. Preprocess permissions (simple lowercasing for this example)
    permission_keywords = set([p.split('.')[-1].lower() for p in declared_permissions])

    # 4. Calculate Jaccard Similarity (the "mismatch" score)
    similarity = jaccard_similarity(review_keywords, permission_keywords)
    print(f"Permission-Behavior Jaccard Similarity: {similarity:.2f}")

    # 5. Calculate final Risk Score (as per the paper's formula)
    # These weights are examples
    permission_weight = 0.9 # e.g., for ACCESS_FINE_LOCATION
    usage_frequency = 0.95 # e.g., this is a common complaint
    
    alpha, beta, gamma = 0.5, 0.3, 0.2
    risk_score = (alpha * nlp_score) + (beta * permission_weight) + (gamma * usage_frequency)
    
    return {
        "risk_category": risk_category,
        "risk_score": risk_score,
        "jaccard_similarity": similarity
    }

if __name__ == '__main__':
    sample_review = "This app tracks my location without asking, it's very creepy."
    sample_permissions = [
        'android.permission.INTERNET',
        'android.permission.ACCESS_FINE_LOCATION',
        'android.permission.CAMERA'
    ]
    
    result = analyze_review_and_permissions(sample_review, sample_permissions)
    print("\n--- Analysis Result ---")
    print(f"Final Risk Score: {result['risk_score']:.2f}")