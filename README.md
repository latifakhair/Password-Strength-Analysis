🛡️ Password Strength Analysis: Data-Driven Weakness Detection Model

This project applies **Data Analysis and Machine Learning (Random Forest Classifier)** to solve the critical cybersecurity problem of weak passwords.

Using $\mathbf{14}$ million compromised credentials from the **RockYou.txt Dataset**, we developed a scientific $\mathbf{Feature\ Engineering}$ pipeline that calculates **Shannon Entropy** and **Character Classes** to determine true password strength.

**Key Outcome:** A functional **Flask web application** and data-backed policies (e.g., mandate 4 classes and reject entropy below 60 bits) to replace outdated, insufficient length requirements.

## 🚀 Quick Start (Demo)

1.  **Install requirements:** `pip install -r requirements.txt`
2.  **Navigate** to `03_Deployment_Flask_Demo/`
3.  **Run the app:** `python app.py`
4.  **Access in browser:** `http://127.0.0.1:5000/`

## ⚙️ Project Structure

* - `Notebooks/`: Jupyter Notebook containing all data cleaning, feature engineering, and model training.
* - `Data/`: Source and labeled feature datasets.
* `Projet_Demo/`: Complete source code for the live web demonstration
