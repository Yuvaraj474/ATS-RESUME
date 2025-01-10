ATS Resume Tracking System 
We have used gemimi api key , LLM model from Google in a conda environment 
using venv with python that helps us to match the Resume for the given Job description and 
give us the matching percentage and the missing keywords

Commands:

conda create -p venv python==3.10 -y

conda activate venv/

pip install -r requirements.txt

streamlit run app.py
