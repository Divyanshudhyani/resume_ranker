import sys
from ranker import ResumeRankerAI

def main():
    """
    Accepts a job description as input, ranks resumes, and handles new resume additions.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py '<job_description>'")
        sys.exit(1)

    job_description = sys.argv[1]  # Get job description from command line
    ranker = ResumeRankerAI("data/Resume.csv")  # Initialize ResumeRankerAI with dataset

    # Add an example of a new resume dynamically (can be replaced with user input)
    new_resume_text = "Experienced software developer with expertise in Python, ML, and AI."
    ranker.add_new_resume(new_resume_text, resume_id="NEW_001")

    # Rank resumes
    ranked_resumes = ranker.rank_resumes(job_description)

    print("\nTop 5 Matching Resumes:")
    print(ranked_resumes.head(50))  # Print top 5 ranked resumes

if __name__ == "__main__":
    main()
