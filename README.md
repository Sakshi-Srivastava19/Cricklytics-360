# 🏏 Cricklytics-360  
### AI-Powered IPL Analytics, Prediction & Insights Dashboard

Cricklytics-360 is a Streamlit-based web application that predicts IPL match outcomes, compares ML models, displays cricket insights, and includes an interactive AI chatbot assistant powered by OpenAI.

---

## 🚀 Features

### 🔮 **Match Prediction**
- Predicts match winner using XGBoost and other ML models  
- Inputs: Teams, Venue, Toss, Batting order, Recent form, etc.  
- Outputs: Probability of winning + explanation  

### 🧠 **Compare ML Models**
- Evaluate Logistic Regression, Random Forest, XGBoost  
- Accuracy, F1-score, Confusion Matrix  
- Visual comparison  

### 📊 **IPL Insights**
Interactive visual analytics:  
- Top venues  
- Wins by team  
- Matches per season  
- Team performance filters  

### 🤖 **Chatbot Assistant**
- IPL Q&A  
- Match analysis  
- Player stats questions  
- Cricket analytics support  

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **XGBoost, Scikit-learn**
- **Pandas & NumPy**
- **Plotly**
- **OpenAI API**

---
## 📁 Project Structure
Cricklytics-360/
│── app.py
│── requirements.txt
│── README.md
│── .streamlit/
│ └── secrets.toml
│── models/
│ └── xgboost_model.json
│── data/
│ ├── processed_matches.csv
│── pages/
│── assets/


---

---

## 🔧 Installation

### 1️⃣ Clone the repo

git clone https://github.com/your-username/Cricklytics-360.git

cd Cricklytics-360

### 2️⃣ Install dependencies

pip install -r requirements.txt


### 3️⃣ Add your API key  
Create:

.streamlit/secrets.toml


OPENAI_API_KEY="your_api_key_here"



### 4️⃣ Run the app  

streamlit run app.py

## 🌐 Deployment (Streamlit Cloud)

1. Go to https://share.streamlit.io  
2. Connect your GitHub  
3. Select the `Cricklytics-360` repo  
4. Choose:
   - **Main file path:** `app.py`
5. Add Secrets:
   - Settings → Secrets → Paste the same:
     ```
     OPENAI_API_KEY="your_api_key_here"
     ```
6. Click **Deploy**

Your app goes live in 30–40 seconds 🎉

---

## 📬 Support

If you face issues, feel free to create an Issue or contact the developer.

---

## 🎉 Enjoy Cricklytics-360!


## 📁 Project Structure

