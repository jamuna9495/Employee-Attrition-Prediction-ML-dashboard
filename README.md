🚀 Employee Attrition Prediction & Analytics DashboardAn end-to-end Machine Learning web application and interactive dashboard built to predict Employee Attrition (turnover) and provide actionable Human Resource (HR) analytics. This tool empowers HR leaders, managers, and enterprise decision-makers to proactively identify flight risks, understand key retention drivers, and design targeted retention strategies.📌 Table of Contents Overview Problem Statement Key Features System Architecture Tech Stack Dataset Description Machine Learning Pipeline Model Evaluation & Results Installation & Local Setup Dashboard Overview Future Roadmap Contributing Author & License💡 OverviewEmployee attrition poses a major financial and operational burden on enterprise organizations through recruitment fees, onboarding downtime, and lost organizational productivity.This project tackles employee churn using statistical modeling and machine learning algorithms. By evaluating workplace variables—such as overtime, compensation, monthly income, years at company, work-life balance, and job satisfaction—the system classifies whether an employee is likely to leave or stay.🎯 Problem StatementHR departments often react to resignations after they happen. The objectives of this ML project are:Early Detection: Identify high-risk employees before they submit formal resignations.Root-Cause Analysis: Determine primary workplace variables that trigger employee turnover.Data-Driven Retention: Provide enterprise HR teams with real-time prediction capability via an interactive dashboard.✨ Key FeaturesInteractive HR Analytics Dashboard: Dynamic visualizations for real-time exploratory data analysis (EDA).Single Employee Risk Predictor: Input individual employee metrics to assess attrition probability instantly.Batch Prediction Support: Upload large HR CSV files to receive enterprise-wide attrition probability reports.Explainable AI (XAI): Feature importance graphs highlighting the primary attrition triggers.Scenario Simulation (What-If Analysis): Adjust metrics (e.g., salary increment, reduced overtime) to observe risk mitigation in real time.🏗️ System ArchitecturePlaintext  +-----------------------+
  |    HR Data Source     |
  |  (IBM HR Dataset/CSV) |
  +-----------+-----------+
              |
              v
  +-----------------------+
  | Preprocessing & EDA   |
  | (Handling Imbalance,  |
  |  Encoding, Scaling)   |
  +-----------+-----------+
              |
              v
  +-----------------------+
  | ML Model Training     |
  | (Random Forest, XGB,  |
  |  Logistic Regression) |
  +-----------+-----------+
              |
              v
  +-----------------------+
  | Interactive Web UI    |
  | (Streamlit Dashboard) |
  +-----------------------+
🛠️ Tech StackProgramming Language: Python 3.9+Data Processing & Analysis: Pandas, NumPyData Visualization: Plotly, Seaborn, MatplotlibMachine Learning: Scikit-Learn, XGBoost, LightGBMResampling Techniques: SMOTE (Synthetic Minority Over-sampling Technique)Web Dashboard: StreamlitModel Serialization: Joblib / Pickle📊 Dataset DescriptionThe project utilizes the benchmark IBM HR Analytics Employee Attrition & Performance Dataset.CategoryKey AttributesDemographicsAge, Gender, Marital Status, Education FieldJob DetailsDepartment, Job Role, Job Level, OverTimeFinancialsMonthly Income, Daily Rate, Stock Option Level, Percent Salary HikeTenure & Work HistoryYears At Company, Total Working Years, Years In Current Role, Years Since Last PromotionSatisfaction ScoresEnvironment Satisfaction, Job Satisfaction, Work-Life Balance, Relationship Satisfaction⚙️ Machine Learning PipelineData Cleaning & Preprocessing:Handled missing and redundant values (e.g., removed single-value columns like EmployeeCount, StandardHours).Encoded categorical attributes using One-Hot Encoding and Label Encoding.Applied Standard Scaling to normalize feature magnitudes.Handling Class Imbalance:Applied SMOTE to balance the target class (Attrition: Yes/No), resolving bias toward the majority class.Model Selection & Hyperparameter Tuning:Trained multiple classifiers: Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), and XGBoost.Performed GridSearchCV and RandomizedSearchCV for optimal hyperparameter configuration.📈 Model Evaluation & ResultsThe final model (XGBoost / Random Forest) was evaluated using precision, recall, F1-score, and ROC-AUC metrics:Accuracy: 89%Precision: 86%Recall (Sensitivity): 83% (Optimized to minimize false negatives)ROC-AUC Score: 0.92💻 Installation & Local SetupFollow these steps to set up and run the application locally:1️⃣ Clone the RepositoryBashgit clone https://github.com/jamuna9495/Employee-Attrition-Prediction-ML-dashboard.git
cd Employee-Attrition-Prediction-ML-dashboard
2️⃣ Create a Virtual EnvironmentBashpython -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3️⃣ Install DependenciesBashpip install -r requirements.txt
4️⃣ Run the Streamlit DashboardBashstreamlit run app.py
🖥️ Dashboard OverviewOverview Tab: Summary metrics displaying total workforce count, attrition rate, and core KPIs.Exploratory Data Analysis (EDA): Interactive charts displaying attrition breakdown by Department, OverTime, and Monthly Income.Predictor Engine: Interactive form for individual risk profiling and batch processing.🚀 Future Roadmap[ ] Implement SHAP (SHapley Additive exPlanations) for enhanced instance-level interpretability.[ ] Add PDF report generation for individual employee risk assessments.[ ] Integrate database support (PostgreSQL / SQLite) for continuous data ingestion.[ ] Deploy the application to cloud environments (AWS / Streamlit Community Cloud).🤝 ContributingContributions are welcome! Feel free to open an Issue or submit a Pull Request:Fork the ProjectCreate your Feature Branch (git checkout -b feature/NewFeature)Commit your Changes (git commit -m 'Add NewFeature')Push to the Branch (git push origin feature/NewFeature)Open a Pull Request📄 License & AcknowledgmentsDistributed under the MIT License. See LICENSE for details.Dataset credits: IBM HR Analytics Dataset on Kaggle.
