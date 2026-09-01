# Predicting Life Expectancy Using Socioeconomic and Health Indicators in African Countries

## AnalystLab Africa Data Science Internship — Week 8 Capstone Project

**Author:** Alvine Lumiti Makutu
**Project Type:** Data Science and Machine Learning Capstone
**Dataset:** World Bank World Development Indicators (WDI)
**Geographic Scope:** African Countries
**Period Covered:** 2000–2025
**Target Variable:** Life Expectancy at Birth, Total (Years)

---

# 1. Introduction

Life expectancy is an important indicator of population health, quality of life, and socioeconomic development. It reflects the overall health conditions of a population and is influenced by several interconnected factors, including healthcare access, economic conditions, education, sanitation, infrastructure, and employment.

African countries experience substantial differences in life expectancy. These differences are associated with variations in healthcare systems, access to basic infrastructure, economic development, education, and living conditions.

This project uses data from the **World Bank World Development Indicators (WDI)** to investigate whether socioeconomic, health, education, infrastructure, population, and temporal indicators can be used to predict life expectancy in African countries.

The project applies exploratory data analysis and machine learning techniques to identify important patterns and develop predictive models. Two regression algorithms, Random Forest and Gradient Boosting, are trained and evaluated using standard regression metrics.

The final model is also deployed through a Streamlit web application, allowing users to enter indicator values and obtain an estimated life expectancy.

---

# 2. Problem Statement

Life expectancy varies considerably across African countries and across different periods. Understanding the factors associated with these variations can provide useful insights for policymakers, researchers, and development organizations.

Traditional analysis can identify relationships between individual indicators and life expectancy, but machine learning provides an opportunity to model multiple socioeconomic and health indicators simultaneously.

The problem addressed in this project is therefore:

> **Can socioeconomic, health, education, infrastructure, population, and temporal indicators from the World Bank be used to predict life expectancy in African countries?**

The project aims to develop a machine learning model capable of estimating life expectancy based on selected World Development Indicators.

---

# 3. Research Question

**Can socioeconomic, health, education, and infrastructure indicators be used to predict life expectancy in African countries?**

---

# 4. Project Objectives

The main objective of this project is to develop and deploy a machine learning model for predicting life expectancy in African countries using World Bank development indicators.

### Specific Objectives

1. Collect and understand relevant data from the World Bank World Development Indicators dataset.
2. Clean and preprocess the selected data for analysis and modelling.
3. Explore relationships between life expectancy and socioeconomic, health, education, and infrastructure indicators.
4. Develop at least two machine learning regression models.
5. Evaluate and compare the performance of the models.
6. Identify the indicators that contribute most to the model's predictions.
7. Develop recommendations based on the findings.
8. Deploy the best-performing model using Streamlit.

---

# 5. Data Collection and Understanding

## 5.1 Data Source

The dataset used in this project is the **World Bank World Development Indicators (WDI)** dataset.

The WDI database provides internationally comparable statistics covering economic, social, health, environmental, and development indicators for countries and regions around the world.

The project used the World Bank's bulk CSV dataset, with the analysis focusing specifically on African countries and the period from **2000 to 2025**.

The original WDI dataset contained:

* **396,970 rows**
* **70 columns**
* Country and indicator metadata
* Annual observations covering 1960–2025

Because the complete WDI dataset contains thousands of indicators, only indicators relevant to life expectancy were selected for this project.

---

# 6. Selected Indicators

The target variable and predictive features were selected using their official World Bank indicator codes.

| Indicator          | Indicator Code      | Role    |
| ------------------ | ------------------- | ------- |
| Life Expectancy    | `SP.DYN.LE00.IN`    | Target  |
| GDP per Capita     | `NY.GDP.PCAP.CD`    | Feature |
| Health Expenditure | `SH.XPD.CHEX.GD.ZS` | Feature |
| Physicians         | `SH.MED.PHYS.ZS`    | Feature |
| Electricity Access | `EG.ELC.ACCS.ZS`    | Feature |
| School Enrollment  | `SE.SEC.ENRR`       | Feature |
| Unemployment       | `SL.UEM.TOTL.ZS`    | Feature |
| Population         | `SP.POP.TOTL`       | Feature |
| Basic Sanitation   | `SH.STA.BASS.ZS`    | Feature |
| Year               | —                   | Feature |

### Target Variable

**Life Expectancy at Birth, Total (Years)** represents the average number of years a newborn is expected to live under current mortality conditions.

### Predictor Variables

The predictor variables represent different dimensions of development:

* **GDP per Capita** represents economic conditions.
* **Health Expenditure** represents investment in healthcare.
* **Physicians** represents healthcare workforce availability.
* **Electricity Access** represents infrastructure development.
* **School Enrollment** represents access to education.
* **Unemployment** represents labour-market conditions.
* **Population** captures demographic scale.
* **Basic Sanitation** represents access to essential public-health infrastructure.
* **Year** captures temporal changes in development and life expectancy.

---

# 7. Geographic Scope

The analysis was restricted to African countries rather than using the entire World Bank dataset.

A list of African country codes was used to filter the selected indicators. This resulted in:

* **54 African countries**
* **486 indicator-country combinations**
* **26 years of observations from 2000 to 2025**

Regional aggregates and non-African countries were excluded from the modelling dataset.

---

# 8. Data Preprocessing

## 8.1 Data Reshaping

The original WDI CSV file stores years as separate columns, creating a wide-format dataset.

The selected data was transformed into long format using a melt operation. The resulting structure contained:

* Country Name
* Country Code
* Indicator Name
* Indicator Code
* Year
* Value

The data was then pivoted so that each country-year combination became one observation and each selected indicator became a feature.

The resulting modelling dataset contained:

**1,404 country-year observations and 12 columns.**

---

# 9. Missing Value Analysis

Missing values were present in several WDI indicators. This is expected because not every country reports every indicator for every year.

The missing-value analysis produced the following results:

| Variable           | Missing Values | Missing Percentage |
| ------------------ | -------------: | -----------------: |
| Physicians         |            798 |             56.84% |
| School Enrollment  |            666 |             47.44% |
| Health Expenditure |            148 |             10.54% |
| Basic Sanitation   |             86 |              6.13% |
| Electricity Access |             70 |              4.99% |
| Life Expectancy    |             54 |              3.85% |
| GDP per Capita     |             32 |              2.28% |
| Unemployment       |             31 |              2.21% |
| Population         |              0 |              0.00% |
| Year               |              0 |              0.00% |
| Country Name       |              0 |              0.00% |
| Country Code       |              0 |              0.00% |

There were **1,404 total country-year observations**, of which **1,350 contained a recorded life expectancy value**.

Observations without a target value were excluded from machine learning because the model cannot be trained or evaluated without a known target.

For the predictor variables, missing values were handled within the machine learning pipelines using **median imputation**. This prevented missing predictor values from causing model training failures while avoiding the need to manually alter the original dataset.

---

# 10. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to understand the distribution of life expectancy, changes over time, relationships between development indicators, and differences between African countries.

The analysis included:

* Distribution analysis
* Time-series analysis
* Scatter plots
* Correlation analysis
* Country-level comparisons

---

## 10.1 Distribution of Life Expectancy

The distribution of life expectancy was examined using a histogram with a kernel density estimate.

The analysis showed substantial variation in life expectancy across the African country-year observations.

This variation is important for machine learning because it provides sufficient differences in the target variable for the model to learn relationships between life expectancy and the selected predictors.

---

## 10.2 Life Expectancy Over Time

Average life expectancy was calculated for each year from 2000 to 2025.

The time-series analysis was used to examine how average life expectancy changed across the study period.

Including **Year** as a model feature allows the machine learning algorithms to account for temporal patterns and broad improvements or changes in life expectancy over the period studied.

---

## 10.3 Relationship Between Life Expectancy and Development Indicators

Scatter plots were created to examine the relationship between life expectancy and:

* GDP per Capita
* Electricity Access
* Basic Sanitation
* Health Expenditure

The visual analysis indicates that life expectancy is associated with broader socioeconomic and infrastructure conditions.

Countries with stronger access to basic infrastructure and sanitation generally tend to have higher life expectancy, while countries with lower levels of development tend to experience lower life expectancy.

However, these relationships should not be interpreted as evidence of direct causation.

---

# 11. Country-Level Life Expectancy Analysis

Average life expectancy was calculated for each African country across the available observations.

## 11.1 Countries With the Highest Average Life Expectancy

| Rank | Country          | Average Life Expectancy |
| ---: | ---------------- | ----------------------: |
|    1 | Tunisia          |                   74.47 |
|    2 | Algeria          |                   74.08 |
|    3 | Seychelles       |                   73.51 |
|    4 | Mauritius        |                   73.26 |
|    5 | Cabo Verde       |                   72.82 |
|    6 | Libya            |                   71.71 |
|    7 | Morocco          |                   71.39 |
|    8 | Egypt, Arab Rep. |                   69.45 |
|    9 | Mauritania       |                   65.56 |
|   10 | Gabon            |                   65.17 |

Tunisia had the highest average life expectancy among the countries in the analysed dataset, with an average of approximately **74.47 years**.

---

## 11.2 Countries With the Lowest Average Life Expectancy

| Rank | Country                  | Average Life Expectancy |
| ---: | ------------------------ | ----------------------: |
|    1 | Central African Republic |                   45.63 |
|    2 | Lesotho                  |                   49.68 |
|    3 | Chad                     |                   50.91 |
|    4 | Nigeria                  |                   51.29 |
|    5 | Eswatini                 |                   52.19 |
|    6 | Somalia, Fed. Rep.       |                   52.32 |
|    7 | South Sudan              |                   52.58 |
|    8 | Sierra Leone             |                   53.62 |
|    9 | Zimbabwe                 |                   54.17 |
|   10 | Mali                     |                   56.36 |

The difference between the highest and lowest country averages demonstrates the considerable variation in life expectancy across African countries.

---

# 12. Machine Learning Model Development

A supervised machine learning approach was used because the project objective was to predict a continuous numerical target: life expectancy in years.

Two regression algorithms were selected:

1. Random Forest Regressor
2. Gradient Boosting Regressor

These algorithms were selected because they can model nonlinear relationships and interactions between socioeconomic and health indicators.

---

# 13. Train-Test Split

The modelling dataset contained **1,350 observations with available life expectancy values**.

The data was divided into:

* **80% training data:** 1,080 observations
* **20% testing data:** 270 observations

The split used:

* `test_size = 0.20`
* `random_state = 42`

The training dataset was used to fit the models, while the test dataset was reserved for evaluating their performance on unseen observations.

---

# 14. Model 1: Random Forest Regressor

The first model was a Random Forest Regressor.

The model was configured with:

* **300 decision trees**
* `random_state = 42`
* Parallel processing using all available CPU cores

Random Forest combines predictions from multiple decision trees. This allows it to capture nonlinear relationships between the development indicators and life expectancy while reducing the risk of relying on a single decision tree.

---

# 15. Model 2: Gradient Boosting Regressor

The second model was a Gradient Boosting Regressor.

The model used:

* **200 estimators**
* Learning rate of **0.05**
* Maximum tree depth of **3**
* `random_state = 42`

Gradient Boosting builds models sequentially, with each new model attempting to improve upon the errors made by previous models.

---

# 16. Model Evaluation

Three evaluation metrics were used:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between the predicted and actual life expectancy values.

A lower MAE indicates better performance.

### Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction errors.

RMSE penalizes larger errors more strongly than MAE. A lower RMSE indicates better performance.

### R² Score

R² measures the proportion of variance in the target variable explained by the model.

A higher R² indicates better explanatory performance.

---

# 17. Model Performance Results

The two models produced the following results on the test dataset:

| Model             |    MAE |   RMSE |     R² |
| ----------------- | -----: | -----: | -----: |
| Random Forest     | 1.2029 | 1.8779 | 0.9323 |
| Gradient Boosting | 1.8773 | 2.4619 | 0.8836 |

---

## 17.1 Best-Performing Model

The **Random Forest Regressor** achieved the best performance across all three evaluation metrics.

Its results were:

* **MAE:** 1.20 years
* **RMSE:** 1.88 years
* **R²:** 0.9323

The R² score indicates that the Random Forest model explained approximately **93.2% of the variance in life expectancy in the test dataset**.

The MAE of approximately **1.20 years** indicates that the model's predictions differed from the actual values by about 1.20 years on average.

Based on these results, the Random Forest model was selected as the final model for deployment.

---

# 18. Feature Importance Analysis

Feature importance was extracted from the Random Forest model to determine which variables contributed most strongly to its predictions.

| Rank | Feature            | Importance |
| ---: | ------------------ | ---------: |
|    1 | Electricity Access |     29.50% |
|    2 | Basic Sanitation   |     27.51% |
|    3 | Year               |     16.18% |
|    4 | Population         |     10.66% |
|    5 | Unemployment       |      5.36% |
|    6 | GDP per Capita     |      3.78% |
|    7 | Health Expenditure |      3.55% |
|    8 | School Enrollment  |      2.04% |
|    9 | Physicians         |      1.41% |

Electricity access was the most influential feature in the Random Forest model, followed closely by basic sanitation.

Together, electricity access and basic sanitation accounted for approximately **57% of the model's feature importance**.

Year was also an important predictor, suggesting that temporal changes in the dataset contributed substantially to the model's predictions.

Population, unemployment, GDP per capita, health expenditure, school enrollment, and physician availability contributed smaller proportions of the model's predictive importance.

It is important to note that **feature importance represents predictive contribution within the model and does not establish causation**. A high feature importance does not mean that changing the feature by itself will necessarily cause a corresponding change in life expectancy.

---

# 19. Key Findings

The analysis produced several important findings.

### 19.1 Life Expectancy Varies Substantially Across African Countries

The country-level analysis revealed significant differences in average life expectancy.

Tunisia had the highest average life expectancy in the analysed dataset at approximately **74.47 years**, while the Central African Republic had the lowest at approximately **45.63 years**.

This represents a difference of almost 29 years.

### 19.2 Infrastructure Is Highly Important for Prediction

Electricity access was the most important predictor in the Random Forest model.

This highlights the importance of infrastructure as a broad development indicator associated with living conditions and population wellbeing.

### 19.3 Sanitation Was Also a Major Predictor

Basic sanitation was the second most important feature.

Access to sanitation is closely related to environmental health and prevention of diseases, making it an important indicator when modelling population-level health outcomes.

### 19.4 Temporal Changes Matter

Year was the third most important feature.

This indicates that changes occurring over time are important for predicting life expectancy, reflecting broader improvements or changes in socioeconomic and health conditions.

### 19.5 Random Forest Outperformed Gradient Boosting

Random Forest achieved lower MAE and RMSE and a higher R² score than Gradient Boosting.

Therefore, Random Forest was selected as the final model.

---

# 20. Recommendations

Based on the analysis, the following recommendations are proposed.

## 20.1 Improve Access to Electricity

African governments and development partners should continue investing in reliable electricity infrastructure, particularly in underserved rural and low-income communities.

Improved electricity access can support healthcare facilities, education, communication, water systems, and other essential services.

## 20.2 Expand Access to Basic Sanitation

Investment in sanitation infrastructure should remain a public-health priority.

Improving access to safe sanitation can contribute to healthier living environments and help reduce exposure to preventable diseases.

## 20.3 Strengthen Health Systems

Although health expenditure and physician availability had lower model importance than infrastructure variables, healthcare capacity remains an important component of population health.

Governments should continue strengthening healthcare infrastructure, workforce availability, preventive healthcare, and access to essential medical services.

## 20.4 Invest in Education

Improving school enrollment and educational opportunities can contribute to long-term socioeconomic development.

Education policies should focus not only on enrollment but also on quality, completion, accessibility, and equity.

## 20.5 Address Economic and Employment Challenges

GDP per capita and unemployment were included as socioeconomic indicators.

Policies that support economic growth, employment creation, poverty reduction, and improved household living standards can contribute to broader improvements in population wellbeing.

## 20.6 Use Data-Driven Policy Planning

The project demonstrates how publicly available development indicators can be combined with machine learning to identify patterns and generate predictions.

Government agencies and development organizations can use similar analytical approaches to support evidence-based planning, while ensuring that predictions are interpreted alongside domain expertise and contextual information.

---

# 21. Model Deployment

The final Random Forest model was deployed using **Streamlit**.

The trained model was saved using Joblib and integrated into a web-based prediction application.

The application allows users to enter values for:

* GDP per Capita
* Health Expenditure
* Physicians
* Electricity Access
* School Enrollment
* Unemployment
* Population
* Basic Sanitation
* Year

After the user submits the values, the application passes the inputs to the trained Random Forest model and displays the predicted life expectancy in years.

The deployment demonstrates the transition from an experimental machine learning notebook to a usable application.

---

# 22. Technologies Used

The project was developed using the following technologies and libraries:

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**
* **Git**
* **GitHub**

---

# 23. Project Structure

The project was organized into a structured machine learning project:

```text
analystlab-week8-wdi-capstone/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── WDI dataset files
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

The raw World Bank dataset files were excluded from Git version control where appropriate because of their size.

---

# 24. Limitations

Despite the strong model performance, several limitations should be considered.

### 24.1 Missing Data

Several indicators contained substantial missing values, particularly physician availability and school enrollment.

Median imputation was used for missing predictor values, but imputation can introduce assumptions that may affect model performance.

### 24.2 Country-Year Data Structure

The dataset contains repeated observations from the same countries across multiple years.

The project used a random train-test split, meaning observations from the same country and nearby years could appear in both training and testing sets.

As a result, the reported performance may be more optimistic than performance on completely unseen countries or future years.

A future version of the project could use a country-level or time-based validation strategy to provide a stricter evaluation.

### 24.3 Feature Importance Does Not Demonstrate Causality

The feature importance results show which variables were useful for prediction by the Random Forest model.

They should not be interpreted as proof that one indicator directly causes changes in life expectancy.

### 24.4 Limited Model Selection

The project focused on two machine learning algorithms: Random Forest and Gradient Boosting.

Additional algorithms and systematic hyperparameter optimization could potentially improve predictive performance.

### 24.5 External Factors

Life expectancy is influenced by many factors that are not included in the selected dataset, including conflict, disease outbreaks, political instability, nutrition, environmental conditions, healthcare quality, and other demographic and social factors.

Therefore, the model should be treated as a predictive analytical tool rather than a complete explanation of life expectancy.

---

# 25. Conclusion

This project investigated whether socioeconomic, health, education, infrastructure, population, and temporal indicators from the World Bank World Development Indicators dataset could be used to predict life expectancy in African countries.

The analysis covered **54 African countries and 1,404 country-year observations from 2000 to 2025**. After accounting for missing target values, **1,350 observations** were available for machine learning.

Two regression models were developed and evaluated: Random Forest and Gradient Boosting.

The **Random Forest Regressor** achieved the strongest performance, with:

* **MAE:** 1.20 years
* **RMSE:** 1.88 years
* **R²:** 0.9323

The model therefore demonstrated strong predictive performance on the held-out test dataset.

Feature importance analysis identified **electricity access, basic sanitation, and year** as the three most influential predictors in the model.

The findings emphasize the importance of considering infrastructure, sanitation, socioeconomic development, healthcare, education, and temporal trends when analysing population health outcomes.

Finally, the trained Random Forest model was successfully deployed through a **Streamlit application**, demonstrating the complete data science workflow from data collection and preprocessing through exploratory analysis, machine learning, evaluation, and deployment.

Overall, the project demonstrates how World Bank open data and machine learning can be combined to produce useful predictive insights into development and population health across African countries.

---

# 26. Dataset Source

The project uses the World Bank **World Development Indicators (WDI)** dataset.

**World Bank — World Development Indicators**

https://datatopics.worldbank.org/world-development-indicators/

The dataset was accessed through the World Bank's publicly available WDI data and bulk download resources.

---

# 27. Project Deliverables

The completed capstone includes:

1. **Jupyter Notebook** containing data collection, preprocessing, EDA, model development, evaluation, and feature importance analysis.
2. **Trained Random Forest model** saved using Joblib.
3. **Streamlit web application** for life expectancy prediction.
4. **README documentation** describing the project and methodology.
5. **Project report** documenting the complete analysis and findings.
6. **GitHub repository** containing the project source code and documentation.

---

# 28. Final Project Summary

| Component                | Result                                  |
| ------------------------ | --------------------------------------- |
| Dataset                  | World Bank World Development Indicators |
| Countries                | 54 African countries                    |
| Period                   | 2000–2025                               |
| Total observations       | 1,404                                   |
| Observations with target | 1,350                                   |
| Training observations    | 1,080                                   |
| Testing observations     | 270                                     |
| Models                   | Random Forest, Gradient Boosting        |
| Best Model               | Random Forest                           |
| Best MAE                 | 1.2029 years                            |
| Best RMSE                | 1.8779 years                            |
| Best R²                  | 0.9323                                  |
| Top Feature              | Electricity Access                      |
| Deployment               | Streamlit                               |
| Model Serialization      | Joblib                                  |

---

**End of Report**
