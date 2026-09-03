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


The confidence value is obtained from the model's predicted probabilities.
🔬 Machine Learning Model

The project uses a Random Forest Classifier.

Random Forest combines multiple decision trees to make a classification prediction. In this project, the target variable is:

hired


Categorical features are converted into numerical representations using LabelEncoder before training. The dataset is divided into training and testing portions using an 80/20 split. 
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
