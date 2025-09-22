import re

def preprocess_text(text):
    """
    Cleans and preprocesses the resume text by:
    - Converting text to lowercase
    - Removing special characters and extra spaces
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z\s]", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text
