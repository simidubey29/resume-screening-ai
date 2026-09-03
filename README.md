🤖 AI Resume Screening System

An AI-powered web application that predicts whether a candidate is likely to be hired based on resume-related information and candidate attributes. The system uses a Machine Learning classification model with a Flask web interface to provide a hiring prediction along with a confidence score.

🚀 Live Demo

🔗 Deployed Application: https://resumeevolutionanalyzer.streamlit.app/

🔗 GitHub Repository:
https://github.com/simidubey29/resume-screening-ai

📌 Project Overview

Recruiters often need to evaluate a large number of candidates based on multiple factors such as education, skills, experience, internships, projects, certifications, and other achievements.

This project automates the initial screening process by taking candidate information as input and using a trained Random Forest Classifier to predict whether the candidate is likely to be hired.

The application displays:

✅ Hiring prediction
📊 Prediction confidence
🧠 Machine Learning-based candidate evaluation
🌐 Simple and user-friendly web interface

Note: This system is intended as a screening/decision-support tool and should not be used as the sole basis for real-world hiring decisions.

✨ Features
🤖 Machine Learning-based candidate screening
🌲 Random Forest Classification
📊 Hiring probability/confidence score
📝 Web-based candidate input form
🔢 Numerical candidate feature processing
🔤 Categorical feature encoding using Label Encoding
🌐 Flask web application
💾 Trained model saved using Pickle
📁 Dataset-based model training
🧠 How It Works

The system follows this workflow:

Candidate Information
        ↓
Data Preprocessing
        ↓
Categorical Feature Encoding
        ↓
Trained Random Forest Model
        ↓
Prediction
        ↓
Hiring Probability / Confidence
        ↓
Result Displayed on Web Interface


The training process loads the resume dataset, separates the target variable (hired) from the input features, encodes categorical values, and trains a Random Forest classifier. The trained model and encoders are then saved for use by the Flask application. {"fallbackMarkdown":"(GitHub
)","reference":{"matched_text":"","prefix":null,"start_idx":2797,"end_idx":2814,"safe_urls":["https://github.com/simidubey29/resume-screening-ai/blob/main/model.py"],"refs":[],"alt":"(GitHub
)","prompt_text":null,"type":"grouped_webpages","status":"done","items":[{"title":"resume-screening-ai/model.py at main · simidubey29/resume-screening-ai · GitHub","url":"https://github.com/simidubey29/resume-screening-ai/blob/main/model.py","attribution":"GitHub","pub_date":null,"snippet":null,"thumbnail_url":"https://images.openai.com/static-rsc-1/P_qUG3cUDcbx40ezNMNCMvlsTzRn90ik7V0sbdgV7l0VyiV6TW0U55pGtnF10DRAGsOHgevtFeQhtKm89PAdEViJ9i4de6DN2W91fm3WYslu5BmePQSX4MlCMw-ewHdqKIC1txGPA61yAUgqej28Sojp71-M9MBzTQaQ52SkahP1rN15EJ4KtodkxrggErdid6x_n5Pxil-mKnJ_koHinw","attribution_segments":null,"supporting_websites":[],"refs":[{"turn_index":1,"ref_type":"view","ref_index":1}],"hue":null,"attributions":null}],"error":null,"fallback_items":null,"style":null},"showLoginRequiredCard":false}

🛠️ Tech Stack
Technology	Purpose
🐍 Python	Core programming language
🌐 Flask	Web application framework
🤖 Scikit-learn	Machine Learning
🌲 Random Forest	Classification algorithm
🐼 Pandas	Data processing
🔢 NumPy	Numerical operations
💾 Pickle	Model serialization
HTML/CSS	Frontend interface
📊 Candidate Features

The model considers multiple candidate-related attributes, including:

Age
CGPA
Number of internships
Number of projects
Programming languages
Certifications
Years of experience
Hackathons
Research papers
Skills score
Soft skills score
Resume length

These features are processed before being passed to the trained model. {"fallbackMarkdown":"(GitHub
)","reference":{"matched_text":"","prefix":null,"start_idx":3552,"end_idx":3580,"safe_urls":["https://github.com/simidubey29/resume-screening-ai/blob/main/app.py","https://github.com/simidubey29/resume-screening-ai/blob/main/model.py"],"refs":[],"alt":"(GitHub
)","prompt_text":null,"type":"grouped_webpages","status":"done","items":[{"title":"resume-screening-ai/app.py at main · simidubey29/resume-screening-ai · GitHub","url":"https://github.com/simidubey29/resume-screening-ai/blob/main/app.py","attribution":"GitHub","pub_date":null,"snippet":null,"thumbnail_url":"https://images.openai.com/static-rsc-1/zgLYOlNGQhBg9pY8cM-My6Wx700-RK9KtLnTCxBpQeATPXgJwhN4wwYmJ7yBlcoNjR1GskNmerQm9WNbQC-pyD4ETatMLDODe05pT6Ss4Rl95RNw4P8MT4Uas1X8AYE4fAUTJKcpe1ly1mASjP1Lazn8oGgtjVS77GZbixrMqGjw3yOPUVNnD2xS2UoBoPr28RaNUTTObCSl-ytp7kM4Pw","attribution_segments":null,"supporting_websites":[{"title":"resume-screening-ai/model.py at main · simidubey29/resume-screening-ai · GitHub","url":"https://github.com/simidubey29/resume-screening-ai/blob/main/model.py","pub_date":null,"snippet":null,"thumbnail_url":"https://images.openai.com/static-rsc-1/P_qUG3cUDcbx40ezNMNCMvlsTzRn90ik7V0sbdgV7l0VyiV6TW0U55pGtnF10DRAGsOHgevtFeQhtKm89PAdEViJ9i4de6DN2W91fm3WYslu5BmePQSX4MlCMw-ewHdqKIC1txGPA61yAUgqej28Sojp71-M9MBzTQaQ52SkahP1rN15EJ4KtodkxrggErdid6x_n5Pxil-mKnJ_koHinw","attribution":"GitHub"}],"refs":[{"turn_index":1,"ref_type":"view","ref_index":0},{"turn_index":1,"ref_type":"view","ref_index":1}],"hue":null,"attributions":null}],"error":null,"fallback_items":null,"style":null},"showLoginRequiredCard":false}

📁 Project Structure
resume-screening-ai/
│
├── dataset/
│   └── resume.csv
│
├── templates/
│   └── index.html
│
├── app.py
├── model.py
├── utils.py
├── requirements.txt
├── model.pkl
├── encoders.pkl
└── README.md

File Description

app.py
Contains the Flask application, candidate input handling, preprocessing, prediction logic, and result display.

model.py
Loads the dataset, preprocesses categorical features, trains the Random Forest classifier, and saves the trained model and encoders. {"fallbackMarkdown":"(GitHub
)","reference":{"matched_text":"","prefix":null,"start_idx":4119,"end_idx":4147,"safe_urls":["https://github.com/simidubey29/resume-screening-ai/blob/main/app.py","https://github.com/simidubey29/resume-screening-ai/blob/main/model.py"],"refs":[],"alt":"(GitHub
)","prompt_text":null,"type":"grouped_webpages","status":"done","items":[{"title":"resume-screening-ai/app.py at main · simidubey29/resume-screening-ai · GitHub","url":"https://github.com/simidubey29/resume-screening-ai/blob/main/app.py","attribution":"GitHub","pub_date":null,"snippet":null,"thumbnail_url":"https://images.openai.com/static-rsc-1/zgLYOlNGQhBg9pY8cM-My6Wx700-RK9KtLnTCxBpQeATPXgJwhN4wwYmJ7yBlcoNjR1GskNmerQm9WNbQC-pyD4ETatMLDODe05pT6Ss4Rl95RNw4P8MT4Uas1X8AYE4fAUTJKcpe1ly1mASjP1Lazn8oGgtjVS77GZbixrMqGjw3yOPUVNnD2xS2UoBoPr28RaNUTTObCSl-ytp7kM4Pw","attribution_segments":null,"supporting_websites":[{"title":"resume-screening-ai/model.py at main · simidubey29/resume-screening-ai · GitHub","url":"https://github.com/simidubey29/resume-screening-ai/blob/main/model.py","pub_date":null,"snippet":null,"thumbnail_url":"https://images.openai.com/static-rsc-1/P_qUG3cUDcbx40ezNMNCMvlsTzRn90ik7V0sbdgV7l0VyiV6TW0U55pGtnF10DRAGsOHgevtFeQhtKm89PAdEViJ9i4de6DN2W91fm3WYslu5BmePQSX4MlCMw-ewHdqKIC1txGPA61yAUgqej28Sojp71-M9MBzTQaQ52SkahP1rN15EJ4KtodkxrggErdid6x_n5Pxil-mKnJ_koHinw","attribution":"GitHub"}],"refs":[{"turn_index":1,"ref_type":"view","ref_index":0},{"turn_index":1,"ref_type":"view","ref_index":1}],"hue":null,"attributions":null}],"error":null,"fallback_items":null,"style":null},"showLoginRequiredCard":false}

utils.py
Contains text-cleaning functionality such as lowercasing and removing unnecessary characters. {"fallbackMarkdown":"(GitHub
)","reference":{"matched_text":"","prefix":null,"start_idx":4260,"end_idx":4277,"safe_urls":["https://github.com/simidubey29/resume-screening-ai/blob/main/utils.py"],"refs":[],"alt":"(GitHub
)","prompt_text":null,"type":"grouped_webpages","status":"done","items":[{"title":"resume-screening-ai/utils.py at main · simidubey29/resume-screening-ai · GitHub","url":"https://github.com/simidubey29/resume-screening-ai/blob/main/utils.py","attribution":"GitHub","pub_date":null,"snippet":null,"thumbnail_url":"https://images.openai.com/static-rsc-1/iCRWQZzLUpgqZFi-IOV4_BTAE-uxnsmS76Yf0Vp9ou38CYVSStFE6hUeOYy_SO9sLhTZyrgEwtAOhDS-_RNsqUeShF6QaweyQpLdxcSqDfaXRTUISvJ8Gc_9N3it7TymEmuF3JnbuH68nZuDd_Zmq9VyZ_i3_SMRRocme6aG4W7rPWRGND0aD5erw_DBd2UqkAzFr12-KaO4ri6S1pbHkA","attribution_segments":null,"supporting_websites":[],"refs":[{"turn_index":1,"ref_type":"view","ref_index":2}],"hue":null,"attributions":null}],"error":null,"fallback_items":null,"style":null},"showLoginRequiredCard":false}

dataset/resume.csv
Contains the candidate dataset used to train the Machine Learning model.

templates/index.html
Provides the web interface for entering candidate information.

model.pkl
Serialized trained Machine Learning model.

encoders.pkl
Saved LabelEncoder objects used to transform categorical input values.

⚙️ Installation
1. Clone the Repository
git clone https://github.com/simidubey29/resume-screening-ai.git
cd resume-screening-ai

2. Create a Virtual Environment
python -m venv venv

3. Activate the Virtual Environment

Windows:

venv\Scripts\activate


Linux / macOS:

source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

🧪 Train the Model

If you want to retrain the Machine Learning model:

python model.py


This trains the Random Forest classifier and generates the required serialized model and encoder files.

▶️ Run the Application

Start the Flask application:

python app.py


The application will run locally at:

http://127.0.0.1:5000/


Open the URL in your browser and enter the candidate information.

🎯 Example Output

The application produces one of the following types of results:

Candidate is likely to be HIRED 🎉
Confidence: 87.42%


or

Candidate is NOT likely to be hired ❌
Confidence: 72.15%


The confidence value is obtained from the model's predicted probabilities. {"fallbackMarkdown":"(GitHub
)","reference":{"matched_text":"","prefix":null,"start_idx":5824,"end_idx":5841,"safe_urls":["https://github.com/simidubey29/resume-screening-ai/blob/main/app.py"],"refs":[],"alt":"(GitHub
)","prompt_text":null,"type":"grouped_webpages","status":"done","items":[{"title":"resume-screening-ai/app.py at main · simidubey29/resume-screening-ai · GitHub","url":"https://github.com/simidubey29/resume-screening-ai/blob/main/app.py","attribution":"GitHub","pub_date":null,"snippet":null,"thumbnail_url":"https://images.openai.com/static-rsc-1/zgLYOlNGQhBg9pY8cM-My6Wx700-RK9KtLnTCxBpQeATPXgJwhN4wwYmJ7yBlcoNjR1GskNmerQm9WNbQC-pyD4ETatMLDODe05pT6Ss4Rl95RNw4P8MT4Uas1X8AYE4fAUTJKcpe1ly1mASjP1Lazn8oGgtjVS77GZbixrMqGjw3yOPUVNnD2xS2UoBoPr28RaNUTTObCSl-ytp7kM4Pw","attribution_segments":null,"supporting_websites":[],"refs":[{"turn_index":1,"ref_type":"view","ref_index":0}],"hue":null,"attributions":null}],"error":null,"fallback_items":null,"style":null},"showLoginRequiredCard":false}

🔬 Machine Learning Model

The project uses a Random Forest Classifier.

Random Forest combines multiple decision trees to make a classification prediction. In this project, the target variable is:

hired


Categorical features are converted into numerical representations using LabelEncoder before training. The dataset is divided into training and testing portions using an 80/20 split. {"fallbackMarkdown":"(GitHub
)","reference":{"matched_text":"","prefix":null,"start_idx":6256,"end_idx":6273,"safe_urls":["https://github.com/simidubey29/resume-screening-ai/blob/main/model.py"],"refs":[],"alt":"(GitHub
)","prompt_text":null,"type":"grouped_webpages","status":"done","items":[{"title":"resume-screening-ai/model.py at main · simidubey29/resume-screening-ai · GitHub","url":"https://github.com/simidubey29/resume-screening-ai/blob/main/model.py","attribution":"GitHub","pub_date":null,"snippet":null,"thumbnail_url":"https://images.openai.com/static-rsc-1/P_qUG3cUDcbx40ezNMNCMvlsTzRn90ik7V0sbdgV7l0VyiV6TW0U55pGtnF10DRAGsOHgevtFeQhtKm89PAdEViJ9i4de6DN2W91fm3WYslu5BmePQSX4MlCMw-ewHdqKIC1txGPA61yAUgqej28Sojp71-M9MBzTQaQ52SkahP1rN15EJ4KtodkxrggErdid6x_n5Pxil-mKnJ_koHinw","attribution_segments":null,"supporting_websites":[],"refs":[{"turn_index":1,"ref_type":"view","ref_index":1}],"hue":null,"attributions":null}],"error":null,"fallback_items":null,"style":null},"showLoginRequiredCard":false}

🔮 Future Improvements

Some possible improvements for future versions include:

📄 Automatic resume PDF/DOCX parsing
🧠 NLP-based resume analysis
🔍 Job description matching
📊 Candidate ranking system
📈 Model accuracy and evaluation dashboard
📋 Resume scoring based on job requirements
🔐 Recruiter authentication
💾 Database integration
📥 Export candidate results to CSV/Excel
☁️ Cloud deployment
⚖️ Bias and fairness analysis
🧠 Advanced models such as XGBoost or transformer-based models
⚠️ Disclaimer

This application is designed for educational and demonstration purposes.

Hiring decisions can be affected by many factors that cannot be completely captured by a Machine Learning model. Therefore, predictions from this application should be treated as decision-support information, not as a final hiring decision.

👨‍💻 Author

Simi Dubey

GitHub:
https://github.com/simidubey29

Project:
https://github.com/simidubey29/resume-screening-ai

⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
