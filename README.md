**🚀 OVERVIEW**


WaitTimePredict is an interactive, AI-driven dashboard that predicts patient wait times at clinics based on dynamic inputs like staffing levels, walk-ins, and appointment data.

The tool is built using Streamlit and machine learning models (Linear Regression and Random Forest), trained on real-world clinic data.


It helps healthcare administrators optimize appointment scheduling and staffing strategies to reduce patient waiting time and improve operational efficiency.

**📊 FEATURES**


📈 Real-time wait time predictions using ML models

🔁 Dynamic inputs via sliders and toggles (e.g., walk-ins, staff count, equipment downtime)

🌡️ Interactive heatmap for identifying congestion vs. ideal booking times

🔄 Model switcher toggle (Linear Regression / Random Forest)

🧠 Visual feature importance from trained models

🌍 Ready for clinics across diverse urban/rural settings

**🧠 TECH STACK**


Frontend: Streamlit

Backend/ML: scikit-learn, Pandas, NumPy

Visualization: Plotly

Packaging: Joblib (for serialized models)




 **HOW TO RUN**

git clone https://github.com/MikaeelMajiet/WaitTimePredict.git
cd WaitTimePredict

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py


