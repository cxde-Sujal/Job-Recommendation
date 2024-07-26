import streamlit as st
import pickle
import re
import nltk
from urllib.parse import quote

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Custom CSS
def set_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
        }
        .main {
            background-image: linear-gradient(to right ,rgba(128, 106, 83, 0.3),rgba(197, 85,198,0.8));
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            
        }
        .title {
            color: #333;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 1.5rem;
        }
        .title span {
            opacity: 0;
            display: inline-block;
            animation: appear 1s forwards;
        }
        @keyframes appear {
            0% { opacity: 0; transform: translateX(-20px); }
            100% { opacity: 1; transform: translateX(0); }
        }
        .buttons {
            display: flex;
            flex-direction: row;
            align-items: center;
            margin-top: 20px;
            gap:20px;
        }
        .buttons .button-link {
            background-color: rgba(3, 3, 22,1);
            color: white;
            text-decoration: none;
            padding: 15px 30px;
            margin: 10px 0;
            border-radius: 10px;
            display: inline-block;
            transition-duration: 0.4s;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

# Web app
def main():
    set_custom_css()
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Job Recommendation App</h1>", unsafe_allow_html=True)

    # st.title("Job Recommendation App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.markdown(f"<h6 class='hey'>Recommended role:{category_name}</h6>", unsafe_allow_html=True)

        # LinkedIn Job Search URL
        # linkedin_search_url = f"https://www.linkedin.com/jobs/search/?keywords={quote(category_name)}"
        # st.markdown(f"[Jobs based on recommended role  on LinkedIn]({linkedin_search_url})", unsafe_allow_html=True)
        linkedin_search_url = f"https://www.linkedin.com/jobs/search/?keywords={quote(category_name)}"
        internshala_search_url = f"https://internshala.com/internships/{quote(category_name)}-internship"
        indeed_search_url = f"https://www.indeed.com/jobs?q={quote(category_name)}"
        naukri_search_url = f"https://www.naukri.com/{quote(category_name)}-jobs"
        st.markdown(f"<h6 class='hey'>Tap below for latest Jobs and Internships on {category_name}:</h6>", unsafe_allow_html=True)
        st.markdown(f"""
                    <div class="buttons">
                        <a class="button-link" href="{linkedin_search_url}" target="_blank">LinkedIn</a>
                        <a class="button-link" href="{internshala_search_url}" target="_blank">Internshala</a>
                        <a class="button-link" href="{indeed_search_url}" target="_blank">Indeed</a>
                        <a class="button-link" href="{naukri_search_url}" target="_blank">Naukri</a>
                    </div>
                """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Python main
if __name__ == "__main__":
    main()

