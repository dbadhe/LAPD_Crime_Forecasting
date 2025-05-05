# Project: Forecasting Crime Patterns in High-Risk Los Angeles Neighborhoods

## Project Overview

This project aims to analyze and forecast crime patterns within the diverse urban landscape of Los Angeles. Given the complex interplay of socioeconomic factors, cultural diversity, and geography in LA, understanding and predicting criminal activities is crucial. The primary goal was to analyze past crime data and build a predictive model to determine the likelihood of a serious (violent) crime occurring based on situational factors.

## Dataset

* **Source:** Kaggle
* **Dataset Used:** Primarily "Crime Data in Los Angeles (2020 to present)". An initial analysis considered data from 2010-2022, but the focus was narrowed due to machine limitations. The analysis presented primarily uses data from 2021 onwards.
* **Size:** The cleaned dataset used for the final analyses contained 68,824 rows and 15 columns. Original data had 276,584 rows and 28 columns.

## Data Cleaning & Preprocessing

Extensive data cleaning and preprocessing steps were undertaken to prepare the dataset for machine learning models:

1.  **Missing Value Handling:**
    * Columns with high percentages of missing values were dropped (e.g., `Crm Cd 2`, `Crm Cd 3`, `Crm Cd 4`, `Cross Street`, `Weapon Used Cd`, `Weapon Desc`, `Mocodes`).
    * Missing `Vict Sex` and `Vict Descent` values were initially handled, with `Vict Sex` rows eventually dropped due to a smaller percentage of missing data. `Vict Descent` was grouped into top categories and 'Other'.
    * Missing `Premis Desc` values were minimal.
    * Numerical features like `Vict Age` had missing or invalid values (e.g., < 0) replaced with the average age.
2.  **Outlier Management:** Ages less than 0 were replaced with the average.
3.  **Attribute Conversion & Consolidation:**
    * **Crime Categories:** 132 crime codes were consolidated into 6 initial categories based on severity (using California Penal Code guidance) and later into 2 categories (0: Non-Serious, 1: Serious) for binary classification. Irrelevant crime code columns were dropped.
    * **Victim Variables:** `Vict Age` was binned using both equal frequency and pre-defined bins. `Vict Descent` categories were consolidated.
    * **Date/Time:** `DATE OCC` was converted to datetime objects to extract features like Day, Weekday Name, Month Name, Year, and Hour.
    * **Location:** Latitude and Longitude were combined into a `LatLon` string column and then label encoded; original Lat/Lon columns were dropped. Textual location fields were generally not used directly.
4.  **Dropping Irrelevant Columns:** Columns deemed irrelevant for the analysis (e.g., textual descriptions, report dates, status) were removed.

## Methodology & Data Analysis

The analysis involved applying and evaluating various machine learning models in several stages:

**Workflow:**
1.  **Initial Analysis (Clean Data):** Applied Naive Bayes and Tree-based models to the cleaned dataset.
2.  **Feature Selection:** Used techniques like MDI (Mean Decrease in Impurity) for tree models and Sequential Feature Selector (forward/backward) for Naive Bayes to identify the most relevant features. Features selected included 'Rpt Dist No', 'Vict Sex', 'Vict Descent', 'Date', 'Month', 'Hour', 'Vict_Age1', 'Day', 'LatLon'.
3.  **SMOTE Sampling:** Addressed class imbalance (more non-serious crimes than serious ones) by using the Synthetic Minority Over-sampling Technique (SMOTE) to create an evenly distributed dataset.

**Models Used:**
* **Naive Bayes:** MultinomialNB, GaussianNB, ComplementNB, BernoulliNB.
* **Tree-Based Models:**
    * Decision Trees (using Gini and Entropy criteria)
    * Random Forests
    * Extremely Randomized Trees (ExtraTrees)
* **Model Tuning & Evaluation:**
    * **Pre-Pruning & Post-Pruning:** Applied to Decision Trees using `GridSearchCV` to optimize parameters (`max_depth`, `min_samples_leaf`, `min_samples_split`) and CCP alpha for post-pruning.
    * **Stratified Sampling:** Used k-fold cross-validation with stratification to ensure representative sampling during training and testing.
    * **Ensemble Learning:**
        * Bagging (Applied to Naive Bayes and used implicitly in Random Forest).
        * Boosting (AdaBoost, XGBoost mentioned conceptually).
        * Voting Classifiers (Hard and Soft Voting).
* **Evaluation Metrics:** Accuracy (Training and Test), Expected Value (using a custom cost matrix penalizing misclassification of serious crimes heavily), Stratified Accuracy, and ROC AUC curves.

## Results

* **Initial Models:** Basic models showed signs of overfitting (high train accuracy, lower test accuracy). GaussianNB performed well among Naive Bayes models initially. ExtraTrees showed promise among tree models.
* **Pruning:** Post-pruning Decision Trees significantly improved test accuracy and expected value compared to unpruned or pre-pruned versions.
* **Feature Selection Impact:** Re-running models after feature selection yielded different performance rankings, with GaussianNB and BernoulliNB showing high expected values.
* **SMOTE Impact:** Applying SMOTE to balance classes led to different results again. Random Forest achieved the highest test accuracy (70.75%) and a good expected value in this phase. BernoulliNB performed poorly after SMOTE.
* **Best Models:** Across the different stages and evaluations (including ROC curves from the final SMOTE phase), **Random Forest** consistently emerged as one of the top-performing models, followed by ensemble methods (like Soft Voting) and potentially pruned/optimized Decision Trees (Gini/Entropy). GaussianNB was also a strong contender, particularly in expected value before SMOTE.

## Conclusion

The project successfully developed machine learning models to predict the occurrence of serious crimes in Los Angeles based on various features related to time, location, and victim demographics. Through iterative steps of data cleaning, feature engineering, model selection, tuning, and addressing class imbalance with SMOTE, the Random Forest model achieved a test accuracy of approximately 70.75%. This indicates a reasonable capability to forecast crime patterns given specific input data.

## Future Work / Potential Improvements

* Utilize the full dataset spanning from 2010 to the present to capture broader trends.
* Incorporate more variables, potentially using label encoding for categorical features instead of dropping them.
* Explore more advanced models like Artificial Neural Networks (ANN) for potentially higher accuracy.

## Dependencies

*(Based on the analysis performed, typical Python libraries for data science would be required)*
* pandas
* numpy
* scikit-learn (for models, preprocessing, metrics, cross-validation, feature selection)
* imblearn (for SMOTE)
* matplotlib / seaborn (for visualizations like ROC curves, feature importance)

## How to Run

*(Specific instructions depend on the code implementation, which was not provided. Generally, the steps would involve:)*
1.  Ensure all dependencies are installed.
2.  Place the required Kaggle dataset (`Crime Data in Los Angeles (2020 to present)`) in the appropriate directory.
3.  Execute the Python script(s) or Jupyter Notebook(s) containing the data cleaning, preprocessing, model training, and evaluation code.