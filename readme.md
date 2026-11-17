# 🏥 Medical Insurance Cost Prediction
**Machine Learning Zoomcamp 2026 - Midterm Project**

## 📖 Problem Description
Predicting medical costs is a critical task for health insurance companies. Accurately estimating how much a patient's medical bills will cost allows companies to set fair premiums, optimize financial planning, and understand risk factors. 

The goal of this project is to build a Machine Learning model that predicts a patient's medical insurance costs based on their demographic and health characteristics. The model is deployed as a web service using Flask and containerized using Docker.

## 📊 The Dataset
The data used in this project is the **Medical Cost Personal Dataset** sourced from Kaggle. 

**Features:**
* `age`: Age of primary beneficiary.
* `sex`: Insurance contractor gender (female / male).
* `bmi`: Body mass index (ideal range is 18.5 to 24.9).
* `children`: Number of children covered by health insurance / Number of dependents.
* `smoker`: Smoking status (yes / no).
* `region`: The beneficiary's residential area in the US (northeast, southeast, southwest, northwest).
* **Target Variable**: `charges` (Individual medical costs billed by health insurance).

*Note: Because financial targets often have a long-tail distribution, a `log1p` transformation was applied to the `charges` during training, and `expm1` is used to reverse it during the final prediction.*

## 🧠 Model Development
Inside the Jupyter Notebook (`notebook.ipynb`), the following steps were taken:
1. **Data Cleaning & EDA**: Standardized string formatting, checked for missing values, and analyzed feature distributions.
2. **Feature Engineering**: Applied Scikit-Learn's `DictVectorizer` to handle categorical variables (One-Hot Encoding).
3. **Model Selection**: 
   * Trained a baseline **Ridge Regression** model.
   * Trained a **Random Forest Regressor** model.
4. **Hyperparameter Tuning**: The Random Forest heavily outperformed the Ridge Regression. It was further tuned by testing various `max_depth` values. A `max_depth=5` was chosen to prevent overfitting while maintaining high accuracy.

## 📁 Repository Structure
* `notebook.ipynb`: Jupyter notebook containing EDA, feature engineering, and model tuning.
* `train.py`: A standalone Python script that trains the final Random Forest model and saves it.
* `model.bin`: The serialized DictVectorizer and trained Random Forest model.
* `predict.py`: A Flask web service that loads the model and serves predictions via a POST endpoint.
* `test.py`: A client script to test the Flask web service by sending a sample patient profile.
* `Pipfile` & `Pipfile.lock`: Pipenv files for managing dependencies and locking the Python environment (v3.11.16).
* `Dockerfile`: Instructions to containerize the web service for production using Gunicorn.

---

## 🚀 How to Run the Project Locally (Using Pipenv)
You can easily run this project locally using `pipenv` without needing Docker.

### 1. Clone the repository
Open your terminal and run:
```bash
git clone https://github.com/subhadeep322/machine-learning-zoomcamp.git
cd machine-learning-zoomcamp

```
## Install dependencies
Ensure you have pipenv installed. Then, install the required packages:
```bash
pipenv install
```
 Start the Web Service
Enter the virtual environment and start the Flask server:
```bash
pipenv shell
python predict.py
```
Test the Service
Open a new, separate terminal window, navigate to the project folder, and run the test script:
```bash
pipenv run python test.py
```
