# Employee-Attrition-Prediction-ML-dashboard
This repository contains a machine learning project aimed at predicting employee attrition using IBM HR Analytics data. The goal is to provide actionable insights for human resources to reduce employee turnover by identifying those at risk of leaving. Using logistic regression as a baseline model, this project seeks to understand which factors drive employee decisions to stay or leave.

Project Objectives
Data Exploration: Load and explore the dataset to understand its structure and features.
Data Preprocessing: Clean the data, encode categorical variables, and scale features to prepare them for model building.
Model Building: Build a logistic regression model to predict employee attrition, serving as a baseline for future improvement.
Result Analysis: Evaluate the model’s performance and visualise feature importance to understand key factors influencing attrition.
Project Structure
Employee_Attrition_Prediction.ipynb: The main Jupyter Notebook containing the project code and explanations.
requirements.txt: A list of required Python packages to run this project.
data/: The dataset folder (not included here due to file size limits, please see below on how to obtain the dataset).
Dataset
The dataset used for this project is from IBM HR Analytics and contains information about employee demographics, job satisfaction, compensation, and more. It aims to identify key factors that predict employee attrition. You can find the dataset on Kaggle.

Technologies Used
Python 3.7+: Main programming language used.
Jupyter Notebook: Environment for developing and documenting the project.
Libraries:
Pandas: Data manipulation and analysis.
NumPy: Numerical computing.
Scikit-Learn: Machine learning algorithms and tools for model building and evaluation.
Matplotlib & Seaborn: Data visualisation.
How to Run the Project
Clone the Repository:

Open a terminal and run:

git clone https://github.com/jamuna9495/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction
Set Up Environment:

It is recommended to create a virtual environment to run the project: python3 -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate

Install Dependencies:

Install the required Python packages using: pip install -r requirements.txt

Download Dataset:

Download the dataset from Kaggle. Save the dataset file (WA_Fn-UseC_-HR-Employee-Attrition.csv) to the data/ folder in this repository.

Run the Jupyter Notebook:

jupyter notebook Employee_Attrition_Prediction.ipynb
