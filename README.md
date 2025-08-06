# employee-attrition-prediction
Predict employee attrition using machine learning

# Employee Attrition Prediction

This project focuses on predicting employee attrition using machine learning techniques. The goal is to analyze employee-related data to identify patterns and trends that can help organizations retain talent and reduce turnover.

## Project Objective

The primary objective of this project is to:
- Build a predictive model that can classify whether an employee is likely to leave the company.
- Understand the key features and factors influencing employee attrition.
- Provide actionable insights for HR and management.

## Machine Learning Approach

The notebook contains a step-by-step pipeline that includes:
- Data loading and exploration
- Data preprocessing and cleaning
- Feature selection and engineering
- Model training and evaluation
- Model performance metrics (e.g., accuracy, precision, recall, F1-score)

## Technologies Used

- Python
- Jupyter Notebook
- Pandas & NumPy (data manipulation)
- Scikit-learn (machine learning)
- Matplotlib & Seaborn (data visualization)

## Dataset Description

The dataset used contains employee information including:
- Demographic data (age, gender, marital status)
- Job-related data (job role, department, years at company)
- Satisfaction levels (job satisfaction, environment satisfaction)
- Performance metrics (performance rating, training time)

Each record in the dataset includes a target label indicating whether the employee has left or stayed with the company.

## Key Features

- **Age**
- **Department**
- **Job Role**
- **Marital Status**
- **Years at Company**
- **Job Satisfaction**
- **Environment Satisfaction**
- **Overtime**
- **Monthly Income**

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors (KNN)

The best-performing model was selected based on precision, recall, and F1-score.

## Model Results

Summary of results will be based on:
- Confusion matrix
- ROC-AUC curve
- Classification report

(You can fill in the final scores here if available.)

## Project Structure

Employee.prediction.2.ipynb
README.me

##  How to Run

1. Clone the repository.
2. Open the Jupyter Notebook.
3. Install necessary packages (see requirements.txt or run `!pip install` commands in the notebook).
4. Run the cells in sequence.

##  Future Improvements

- Deploy the model as a web application using Flask or Streamlit.
- Use GridSearchCV for hyperparameter tuning.
- Add more advanced ensemble methods (e.g., XGBoost, LightGBM).

##  Real-World Relevance

Understanding why employees leave can help organizations:
- Reduce recruitment and training costs
- Improve workplace satisfaction
- Develop better retention strategies

## References

- IBM HR Analytics Employee Attrition & Performance Dataset
- Scikit-learn documentation
- Kaggle and related HR analytics research papers

## License

This project is open-source and available for use under the MIT License.
