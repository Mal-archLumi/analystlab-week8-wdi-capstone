# Predicting Life Expectancy Using Socioeconomic and Health Indicators in African Countries

## AnalystLab Africa — Week 8 Capstone Project

### Project Overview

This project uses the World Bank's **World Development Indicators (WDI)** dataset to investigate whether socioeconomic, health, education, infrastructure, population, and temporal indicators can be used to predict life expectancy across African countries.

A machine learning regression pipeline was developed and deployed as an interactive **Streamlit** application.

### Research Question

> Can socioeconomic, health, education, and infrastructure indicators be used to predict life expectancy in African countries?

### Objectives

* Collect and prepare World Bank WDI data.
* Focus the analysis on African countries.
* Perform exploratory data analysis.
* Identify relationships between development indicators and life expectancy.
* Train and compare machine learning regression models.
* Evaluate model performance using MAE, RMSE, and R².
* Identify the most influential predictors.
* Deploy the best-performing model using Streamlit.

---

## Dataset

The project uses the World Bank **World Development Indicators** dataset.

Source: https://datatopics.worldbank.org/world-development-indicators/

The analysis covers **54 African countries** and the period **2000–2025**.

### Selected Indicators

| Indicator          | Code                |
| ------------------ | ------------------- |
| Life Expectancy    | `SP.DYN.LE00.IN`    |
| GDP per Capita     | `NY.GDP.PCAP.CD`    |
| Health Expenditure | `SH.XPD.CHEX.GD.ZS` |
| Physicians         | `SH.MED.PHYS.ZS`    |
| Electricity Access | `EG.ELC.ACCS.ZS`    |
| School Enrollment  | `SE.SEC.ENRR`       |
| Unemployment       | `SL.UEM.TOTL.ZS`    |
| Population         | `SP.POP.TOTL`       |
| Basic Sanitation   | `SH.STA.BASS.ZS`    |

---

## Methodology

The project followed these stages:

1. **Data Collection**

   * Downloaded World Bank WDI data.
   * Selected relevant socioeconomic and health indicators.

2. **Data Cleaning**

   * Restricted the dataset to African countries.
   * Selected observations from 2000–2025.
   * Reshaped the WDI data from wide format into country-year observations.
   * Investigated missing values.

3. **Exploratory Data Analysis**

   * Examined life expectancy distributions.
   * Analysed changes over time.
   * Investigated relationships between life expectancy and selected indicators.
   * Generated a correlation matrix.
   * Compared average life expectancy across countries.

4. **Machine Learning**

   * Formulated the task as a supervised regression problem.
   * Used median imputation for missing numerical predictor values.
   * Trained Random Forest and Gradient Boosting regressors.

5. **Evaluation**

   * Mean Absolute Error (MAE)
   * Root Mean Squared Error (RMSE)
   * R²

6. **Deployment**

   * Selected the best-performing model.
   * Saved the trained model using Joblib.
   * Developed an interactive Streamlit application.

---

## Dataset Summary

After preprocessing:

* **African countries:** 54
* **Country-year observations:** 1,404
* **Observations with life expectancy:** 1,350
* **Training observations:** 1,080
* **Testing observations:** 270

Missing values were present in several indicators, particularly Physicians and School Enrollment. Instead of removing large numbers of observations, missing numerical values were handled using median imputation within the machine learning pipeline.

---

## Model Performance

| Model             |       MAE |      RMSE |         R² |
| ----------------- | --------: | --------: | ---------: |
| **Random Forest** | **1.203** | **1.878** | **0.9323** |
| Gradient Boosting |     1.877 |     2.462 |     0.8836 |

### Best Model

The **Random Forest Regressor** achieved the best performance across all three evaluation metrics.

An R² of **0.9323** indicates that the model explains approximately **93.2% of the variation in life expectancy in the test set**.

The MAE of approximately **1.20 years** means that predictions differed from observed life expectancy by about 1.20 years on average.

---

## Feature Importance

The Random Forest model identified the following features as the most influential:

| Feature            | Importance |
| ------------------ | ---------: |
| Electricity Access |     29.50% |
| Basic Sanitation   |     27.51% |
| Year               |     16.18% |
| Population         |     10.66% |
| Unemployment       |      5.36% |
| GDP per Capita     |      3.78% |
| Health Expenditure |      3.55% |
| School Enrollment  |      2.04% |
| Physicians         |      1.41% |

Electricity access and basic sanitation were the two most influential predictors in the trained Random Forest model.

Feature importance represents predictive contribution within the model and should not be interpreted as evidence of causation.

---

## EDA Highlights

Average life expectancy varied substantially across African countries.

The highest average life expectancy values in the dataset were observed for:

* Tunisia — 74.47 years
* Algeria — 74.08 years
* Seychelles — 73.51 years
* Mauritius — 73.26 years
* Cabo Verde — 72.82 years

The lowest average values included:

* Central African Republic — 45.63 years
* Lesotho — 49.68 years
* Chad — 50.91 years
* Nigeria — 51.29 years
* Eswatini — 52.19 years

These differences demonstrate substantial variation in population health outcomes across the continent.

---

## Recommendations

1. **Improve electricity infrastructure**
   Electricity access was the most influential predictor in the Random Forest model. Expanding reliable electricity infrastructure can support healthcare, education, sanitation, and broader socioeconomic development.

2. **Strengthen sanitation infrastructure**
   Basic sanitation was another highly influential predictor. Investment in sanitation and hygiene infrastructure should remain an important public health priority.

3. **Improve development data availability**
   Significant missingness was observed in some indicators, particularly physician availability and school enrollment. More consistent data collection would improve future analyses.

4. **Use data-driven planning**
   Development indicators can be used to identify patterns and potential areas requiring greater investment in health and infrastructure.

5. **Monitor development over time**
   The Year feature showed substantial predictive importance, highlighting the importance of monitoring changes in development and health indicators over time.

---

## Streamlit Application

The trained Random Forest model is deployed through an interactive Streamlit application.

Users can enter socioeconomic, health, infrastructure, education, population, and year-related indicators and receive a predicted life expectancy.

### Run Locally

Clone the repository:

```bash
git clone https://github.com/Mal-archLumi/analystlab-week8-wdi-capstone.git
cd analystlab-week8-wdi-capstone
```

Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
analystlab-week8-wdi-capstone/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── WDICSV.csv
│
├── models/
│   └── life_expectancy_model.pkl
│
├── notebooks/
│   └── week8_life_expectancy_prediction.ipynb
│
├── report/
│   └── week8_capstone_report.md
│
└── src/
```

The raw WDI dataset is excluded from version control where appropriate because of its size. The dataset can be obtained from the official World Bank WDI source.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook
* Git & GitHub

---

## Project Repository

GitHub: https://github.com/Mal-archLumi/analystlab-week8-wdi-capstone.git

## Internship

**AnalystLab Africa — Data Science Internship**

Week 8 Capstone Project.
