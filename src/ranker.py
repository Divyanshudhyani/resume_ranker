import pandas as pd
from sentence_transformers import SentenceTransformer, util
from preprocess import preprocess_text

class ResumeRankerAI:
    def __init__(self, resume_file):
        """
        Loads resumes, preprocesses them, and initializes the Sentence Transformer model.
        """
        self.df = pd.read_csv(resume_file)  # Load dataset
        self.df['Processed_Resume'] = self.df['Resume_str'].apply(preprocess_text)  # Preprocess resumes

        # Load pretrained Sentence Transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Compute embeddings for all resumes in the dataset
        self.df['Embedding'] = self.df['Processed_Resume'].apply(
            lambda x: self.model.encode(x, convert_to_tensor=True)
        )

    def rank_resumes(self, job_description):
        """
        Ranks resumes based on their similarity to the job description.
        Returns a DataFrame with the top matches.
        """
        # Preprocess and compute embedding for the job description
        job_description = preprocess_text(job_description)
        job_embedding = self.model.encode(job_description, convert_to_tensor=True)

        # Calculate cosine similarity between job description and resumes
        self.df['Score'] = self.df['Embedding'].apply(
            lambda x: util.pytorch_cos_sim(job_embedding, x).item()
        )

        # Sort resumes by similarity score
        ranked_resumes = self.df.sort_values(by='Score', ascending=False)
        return ranked_resumes[['ID', 'Resume_str', 'Score']]

    def add_new_resume(self, resume_text, resume_id):
        """
        Adds a new resume dynamically to the system.
        """
        processed_resume = preprocess_text(resume_text)
        embedding = self.model.encode(processed_resume, convert_to_tensor=True)

        # Append the new resume to the DataFrame
        new_entry = {
            'ID': resume_id,
            'Resume_str': resume_text,
            'Processed_Resume': processed_resume,
            'Embedding': embedding,
            'Score': None,  # Score will be computed when ranking
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_entry])], ignore_index=True)
