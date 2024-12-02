# otr_ln_master_table

This project aims to create a master table for OTR and LN data, including the LexisNexis Data.

## 11/18/2024 8:30 AM

### Project Setup

The following steps were taken to set up the project:

1. **Project Creation and GitHub Connection:** The project was created and connected to a GitHub repository for version control and collaboration.
2. **Virtual Environment Creation:** A virtual environment was created to isolate project dependencies and ensure reproducibility.
3. **Project Organization:** The project directory was organized with the following folders:
    - **datasources:** This folder stores the original BCG datasets.
    - **notebooks:** This folder contains Jupyter notebooks used for data exploration, analysis, and visualization.
    - **code_resources:** This folder stores any scripts or modules developed for the project.
4. **Data Ingestion:** The original BCG datasets were added to the `datasources` folder.
5. **Branching and Merging:** A new branch was created for development, and after the initial setup, it was merged to the main branch.

### Future Work

- Develop data cleaning and preprocessing scripts.
- Create a master table by merging and transforming the BCG datasets.
- Develop data analysis and visualization notebooks.
- Document the data sources and the master table schema.

### Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your fork.
5. Submit a pull request to the main branch.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## 11/18/2024 11:34 AM

### Data Sources

The LexisNexis data used in this project comes from two main sources:

1. **BCG's Original OTR Adjudication Model:** This model, found in the Dataiku project [OTR_ADJUDICATION_RISK/](https://dataiku-dss.prod-us-west-2.data.wexapps.com/projects/OTR_ADJUDICATION_RISK/), utilizes two LexisNexis files:
    - `data_lexisnexis.csv`: Contains LN information for applications before 2022, including 53,468 rows and the following columns: `name`, `b2bcnt2y`, `model2score`, and `sbfecardcount`.
    - `data_lexisnexis_2022.csv`: Contains LN information for applications in 2022, including 112,523 rows and the following columns: `accountnumber`, `b2bcnt2y`, `model2score`, and `sbfecardcount`.

2. **New LexisNexis Files:** These files, documented in the Confluence page [LexisNexis One-time Data Directory for NAM and OTR Adjudication Models](https://wexinc.atlassian.net/wiki/spaces/RDS/pages/154849771521/LexisNexis+One-time+Data+Directory+for+NAM+and+OTR+Adjudication+Models), are provided as replacements for the BCG files:
    - `final_wex_11529_sba21_sbfe_sbfescore_pr3.csv`: Covers the period from 01/2021 to 12/2021, joined using either `CREATED_DATE` or `DATE_DECISIONED_C`. Contains `Accountnumber` (ID) and `historydateyyyymm`.
    - `final_wex_11795_otradjud_sba21_sbfe_sbfescore.csv`: Covers the period from 01/2022 to 05/2023, joined using either `CREATED_DATE` or `DATE_DECISIONED_C`. Contains `Accountnumber` (ID) or `NAME` and `historydateyyyymm`.

### Data Validation and Reconciliation

The goal of this project is to:

1. **Create a Master Table:** Integrate the new LexisNexis files into the OTR master table.
2. **Validate Application Counts:** Compare the number of applications in the new files with the corresponding periods in the BCG files to ensure consistency.
3. **Verify Feature Values:** Compare the values of the shared feature `sbfecardcount` between the new and BCG files to ensure data integrity.

This validation process will help identify any discrepancies between the new and old LexisNexis data, ensuring the accuracy and reliability of the OTR master table.

## LexisNexis Data Validation: Pre-2022 Data

This section details the validation process comparing the new and current LexisNexis datasets for applications before 2022.

### Datasets

* **New Data (2021):** `final_wex_11529_sba21_sbfe_sbfescore_pr3.csv`
* **Current Data (2021):** `data_lexisnexis.csv` (from BCG's original OTR adjudication model)

### Comparison Results

The comparison focused on ensuring consistency in terms of rows, null values, data types, and shared feature values.

**1. Structural Comparison:**

* **Rows:** Both datasets contain 52,468 rows.
* **Columns:** The new dataset has 21 columns, while the current dataset has 4 columns.
* **Column Names:** The new dataset includes additional features not present in the current dataset. Both share the following columns: `name`, `b2bcnt2y`, `model2score`, and `sbfecardcount`.
* **Data Types:** Data types for shared columns are consistent across both datasets.

**2. Null Value Comparison:**

* Both datasets exhibit the same number of null values for the shared columns (`b2bcnt2y` and `sbfecardcount`).

**3. Value Comparison:**

* A merge operation was performed to compare the values of shared columns.
* `b2bcnt2y`: 52,455 rows have identical values, while 13 rows have differing values.
* `model2score`: All 52,468 rows have identical values.
* `sbfecardcount`: 52,455 rows have identical values, while 13 rows have differing values.

### Conclusion

The comparison reveals a high degree of concordance between the new and current LexisNexis datasets for applications before 2022. Minor discrepancies in `b2bcnt2y` and `sbfecardcount` values require further investigation to determine the source and potential impact on the master table.



## LexisNexis Data Validation: 2022 Data

This section details the validation process comparing the new and current LexisNexis datasets for applications in 2022.

### Datasets

* **New Data (2022):** `final_wex_11795_otradjud_sba21_sbfe_sbfescore.csv`
* **Current Data (2022):** `data_lexisnexis_2022.csv` (from BCG's original OTR adjudication model)

### Comparison Results

The comparison focused on ensuring consistency in terms of rows, null values, data types, and shared feature values.

**1. Structural Comparison:**

* **Rows:** Both datasets contain 112,523 rows.
* **Columns:** The new dataset has 20 columns, while the current dataset has 4 columns.
* **Column Names:** The new dataset includes additional features not present in the current dataset. Both share the following columns: `accountnumber`, `b2bcnt2y`, `model2score`, and `sbfecardcount`.
* **Data Types:** Data types for shared columns are consistent across both datasets.

**2. Null Value Comparison:**

* Both datasets exhibit the same number of null values for the shared columns (`b2bcnt2y` and `sbfecardcount`).

**3. Value Comparison:**

* A merge operation was performed to compare the values of shared columns.
* `b2bcnt2y`: 112,493 rows have identical values, while 30 rows have differing values.
* `model2score`: All 112,523 rows have identical values.
* `sbfecardcount`: 112,493 rows have identical values, while 30 rows have differing values.

### Conclusion

The comparison reveals a high degree of concordance between the new and current LexisNexis datasets for applications in 2022. Minor discrepancies in `b2bcnt2y` and `sbfecardcount` values require further investigation. However, the overall consistency suggests that the new dataset can effectively replace the original data provided by BCG.

This validation satisfies the requirements outlined in the new flow proposal ([link to Google Docs presentation](https://docs.google.com/presentation/d/1ePdwrILsn0MJBZEAdKstYTfU5C5lgHVK4PGMPy3aiPA/edit)), enabling the retraining of the OTR adjudication model with the updated LexisNexis data.

The validation can be found in the notebook "1_validate_initial_files.ipynb".

## 11/18/2024 4:30 PM

### OTR Adjudication Model Update - LexisNexis Data Integration

This section outlines the changes made to the OTR Adjudication Model in Dataiku 
([flow link](https://dataiku-dss.prod-us-west-2.data.wexapps.com/projects/OTR_ADJUDICATION_RISK/flow/)) 
following the benchmark analysis and integration of new LexisNexis base files.

### Key Changes:

1. **LexisNexis Data Source Update:**
    - The existing LexisNexis base files were replaced with new versions.
    - The processing Python recipe in the Dataiku flow was updated to accommodate the new file paths and structures.
    - Connections to the older LexisNexis files were removed.

2. **New Column Integration:**
    - A new column, `sbfeaccountcount`, was added to the following:
        - `mmconfig` files for each data source.
        - `COLUMNS_LOADED_LN`, `COLUMNS_LOADED_LN_NEW`, and `COLUMNS_LOADED_LN_SINCE_DEC_2023` lists in the processing recipe.
        - `model_monitoring_preprocess_ln_since_2023` function.
        - `OUTPUT_COLUMNS_PREPROCESSED_SAMPLE` object in `mmconfig.py`.

3. **OTR_ADJ_RISK_PREPROCESSED_SAMPLE_MODEL_MONITORING Table Update:**
    - The `OTR_ADJ_RISK_PREPROCESSED_SAMPLE_MODEL_MONITORING` table now includes the `sbfeaccountcount` column. This enables applying segment filters based on this new attribute.

### Column Definitions:

Here's a breakdown of the relevant columns extracted from the LexisNexis datasets:

**Original LexisNexis Dataset:**

```python
COLUMNS_LOADED_LN = [
    "name",
    "b2bcnt2y",
    "model2score",
    "sbfecardcount",
    "sbfeaccountcount",  # New column added
]

COLUMNS_LOADED_LN_NEW = [
    "accountnumber",
    "b2bcnt2y",
    "model2score",
    "sbfecardcount",
    "sbfeaccountcount",  # New column added
]

COLUMNS_LOADED_LN_SINCE_DEC_2023 = [
    "ID",
    "NAME",
    "LEXIS_NEXIS_SBFE_MODEL_SCORE_C",
    "MODEL_2_SCORE_C",
    "B_2_B_CNT_2_Y_C",
    "LEXIS_NEXIS_SBFE_TRADE_COUNT_C",
    "SBFE_ACCOUNT_COUNT_C",  # New column added
]
```


## 11/19/2024 8:20 AM

### OTR Adjudication Model Retraining with New Segmentation

This section details the ongoing efforts to retrain the OTR Adjudication Model with a new segmentation proposal. The goal of this retraining is to improve the model's performance and predictive power by incorporating new data sources and refining the segmentation strategy.

**Retraining Methodology:**

To ensure a smooth transition and minimize disruption to the existing workflow, the retraining process is being conducted in a dedicated Python notebook. This allows for experimentation and validation of the new approach before fully integrating it into the production environment.

- **Flow Proposal Notebook:** [Link to notebook](https://dataiku-dss.prod-us-west-2.data.wexapps.com/projects/OTR_ADJUDICATION_RISK/notebooks/jupyter/Flow%20proposal/)

This notebook outlines the proposed changes to the model flow, including the new segmentation logic, data preprocessing steps, and model training parameters. It serves as a working document for developing and evaluating the new model.

**Flow Proposal Deck:**

A comprehensive flow proposal deck has been created to document the rationale behind the new segmentation approach and the expected benefits. This deck will be continuously updated to reflect the findings and insights gained during the retraining process.

- **Flow Proposal Deck:** [Link to deck](https://docs.google.com/presentation/d/1ePdwrILsn0MJBZEAdKstYTfU5C5lgHVK4PGMPy3aiPA/edit#slide=id.g2f8f80afe75_0_90)

**Data Enhancements and Feature Engineering:**

To further enhance the model's performance, two key data enhancements are being implemented:

1. **Incorporating SBFEHITINDEX:**

   - The `sbfehitindex` feature, derived from LexisNexis datasets, provides valuable information about the applicant's risk profile. This feature has been integrated into the model's input variables to improve the granularity and accuracy of the segmentation.

   - To ensure consistency across different LexisNexis data versions, the corresponding column names (`sbfehitindex`, `SBFE_HIT_INDEX_C`) have been mapped and included in the relevant data loading and preprocessing steps.

2. **Integrating Precise Funding Type Data:**

   - Previously, the model relied on approximated funding type values derived from built query views. This limitation has been addressed by integrating precise funding type information directly from the source table (`prep.salesforece_owner.onlineapplication_c`).

   - This enhancement ensures that the model has access to accurate and granular funding type data, leading to more informed and reliable predictions.

**Configuration and Code Updates:**

To accommodate the new data and features, several updates have been made to the model's configuration files and codebase:

- **Configuration Files:** The `COLUMNS_LOADED_LN`, `COLUMNS_LOADED_LN_NEW`, and `COLUMNS_LOADED_LN_SINCE_DEC_2023` lists in the configuration files have been updated to include the `sbfehitindex` (or its equivalent) column.

- **Python Scripts:** The `model_monitoring_preprocess_ln_since_2023` function within the `m_m_sample.py` file has been modified to incorporate the `sbfehitindex` feature into the data preprocessing pipeline.


## 11/19/2024 3:40 PM

### Data Source Considerations

This section outlines considerations and potential issues related to data sources used in the project.

#### Funding Type Source

**Issue:**  We are considering moving to a new source for `funding_type` data (`PREP.SALESFORCE_OWNER.ONLINEAPPLICATION__C`). However, this table seems to have stopped pulling information in September 2024.

**Analysis:**

* The current `RISK_ANALYTICS.FINCRIMES.sf_online_application_c` table has data from 2011-06-08 to 2024-09-22.
* The proposed `PREP.SALESFORCE_OWNER.ONLINEAPPLICATION__C` table has the same date range, indicating a potential issue with data freshness.

**Recommendation:**

* **Do not move to the new source** until the data update issue is resolved.
* **Continue using the current query** as it still appears to be updated and correctly detects child-funded applications.
* **Confirm with the team** before proceeding with any changes to the funding type source.

**Supporting Queries:**

```sql
SELECT MIN(CREATED_DATE), MAX(CREATED_DATE)
FROM RISK_ANALYTICS.FINCRIMES.sf_online_application_c
LIMIT 10;

SELECT MIN(CREATEDDATE), MAX(CREATEDDATE)
FROM PREP.SALESFORCE_OWNER.ONLINEAPPLICATION__C
LIMIT 10;
```

### `sbfehitindex_ln` Variable

**Issue:** We are considering including the `sbfehitindex_ln` variable, which is already added to the data. We need to confirm if this inclusion causes any information gaps.

**Analysis:**

* An analysis of `SRC_RISK_FRAUD_SUB.RISK_FRAUD.OTR_ADJ_RISK_PREPROCESSED_SAMPLE_MODEL_MONITORING` shows good coverage for `sbfehitindex_ln` across different decision years.
* There are no significant losses of information observed when including this variable.

**Recommendation:**

* **Proceed with including the `sbfehitindex_ln` variable** as it does not appear to introduce significant data gaps.


```sql
SELECT 
    LEFT(oarpsmm."decision_date", 4) AS decision_year, 
    COUNT(oarpsmm."application_number") AS num_applications, 
    SUM(CASE WHEN oarpsmm."booked" = TRUE THEN 1 ELSE 0 END) AS num_booked_applications, 
    SUM(CASE WHEN oarpsmm."booked" = TRUE AND oarpsmm."sbfecardcount_ln" IS NOT NULL THEN 1 ELSE 0 END) AS num_booked_applications_sbfecardcount_available, 
    SUM(CASE WHEN oarpsmm."booked" = TRUE AND oarpsmm."sbfecardcount_ln" IS NOT NULL AND oarpsmm."sbfeaccountcount_ln" IS NOT NULL THEN 1 ELSE 0 END) AS num_booked_applications_sbfecardcount_available_sbfeaccountcount_available, 
    SUM(CASE WHEN oarpsmm."booked" = TRUE AND oarpsmm."sbfecardcount_ln" IS NOT NULL AND oarpsmm."sbfeaccountcount_ln" IS NOT NULL AND oarpsmm."sbfehitindex_ln" IS NOT NULL THEN 1 ELSE 0 END) AS num_booked_applications_sbfecardcount_available_sbfeaccountcount_available_sbfehitindex_available,
    SUM(CASE WHEN oarpsmm."booked" = FALSE THEN 1 ELSE 0 END) AS num_declined_applications, 
    SUM(CASE WHEN oarpsmm."booked" = FALSE AND oarpsmm."sbfecardcount_ln" IS NOT NULL THEN 1 ELSE 0 END) AS num_declined_applications_sbfecardcount_available, 
    SUM(CASE WHEN oarpsmm."booked" = FALSE AND oarpsmm."sbfecardcount_ln" IS NOT NULL AND oarpsmm."sbfeaccountcount_ln" IS NOT NULL THEN 1 ELSE 0 END) AS num_declined_applications_sbfecardcount_available_sbfeaccountcount_available, 
    SUM(CASE WHEN oarpsmm."booked" = FALSE AND oarpsmm."sbfecardcount_ln" IS NOT NULL AND oarpsmm."sbfeaccountcount_ln" IS NOT NULL AND oarpsmm."sbfehitindex_ln" IS NOT NULL THEN 1 ELSE 0 END) AS num_declined_applications_sbfecardcount_available_sbfeaccountcount_available_sbfehitindex_available
FROM SRC_RISK_FRAUD_SUB.RISK_FRAUD.OTR_ADJ_RISK_PREPROCESSED_SAMPLE_MODEL_MONITORING AS oarpsmm
GROUP BY decision_year
ORDER BY decision_year;
```

| DECISION_YEAR | NUM_APPLICATIONS | NUM_BOOKED_APPLICATIONS | NUM_BOOKED_APPLICATIONS_SBFECARDCOUNT_AVAILABLE | NUM_BOOKED_APPLICATIONS_SBFECARDCOUNT_AVAILABLE_SBFEACCOUNTCOUNT_AVAILABLE | NUM_BOOKED_APPLICATIONS_SBFECARDCOUNT_AVAILABLE_SBFEACCOUNTCOUNT_AVAILABLE_SBFEHITINDEX_AVAILABLE | NUM_DECLINED_APPLICATIONS | NUM_DECLINED_APPLICATIONS_SBFECARDCOUNT_AVAILABLE | NUM_DECLINED_APPLICATIONS_SBFECARDCOUNT_AVAILABLE_SBFEACCOUNTCOUNT_AVAILABLE | NUM_DECLINED_APPLICATIONS_SBFECARDCOUNT_AVAILABLE_SBFEACCOUNTCOUNT_AVAILABLE_SBFEHITINDEX_AVAILABLE |
|---|---|---|---|---|---|---|---|---|---|
| 2019 | 34856 | 19131 | 10001 | 5725 | 0 | 0 | 0 | 0 | 0 |
| 2020 | 46245 | 26398 | 19847 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2021 | 53416 | 33758 | 33291 | 33291 | 33291 | 19658 | 19163 | 19163 | 19163 |
| 2022 | 83652 | 51027 | 51027 | 51027 | 51027 | 32625 | 32603 | 32603 | 32603 |
| 2023 | 63778 | 37504 | 19591 | 15767 | 15767 | 26274 | 13885 | 12920 | 12920 |
| 2024 | 40130 | 24819 | 15857 | 11707 | 11707 | 15311 | 9773 | 7941 | 7941 |

## 11/20/2024 9:40 AM

## Retraining of the OTR Adjudication Model

This document outlines the segmentation strategy used for retraining the OTR adjudication model.

### Segmentation Logic

After identifying the necessary columns, we defined the following segments for retraining the model:

```python
conditions = [
    (lambda df: (df['pg_required_c'] == True) & (df['sbfeaccountcount_ln'] >= 1) & (df['sbfehitindex_ln'] >= 2) & (df['fico_score'].notnull())),
    (lambda df: (df['pg_required_c'] == True) & (df['sbfeaccountcount_ln'] >= 1) & (df['sbfehitindex_ln'] >= 2) & (df['fico_score'].isnull())),
    (lambda df: (df['pg_required_c'] == True) & (df['b2bcnt2y_ln'] >= 1) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].notnull())),
    (lambda df: (df['pg_required_c'] == True) & (df['b2bcnt2y_ln'] >= 1) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].isnull())),
    (lambda df: (df['pg_required_c'] == True) & ((df['sbfeaccountcount_ln'] < 1) | (df['sbfeaccountcount_ln'].isnull())) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & ((df['sbfehitindex_ln'] <= 1) | (df['sbfehitindex_ln'].isnull())) & (df['fico_score'].notnull())),
    (lambda df: (df['pg_required_c'] == True) & ((df['sbfeaccountcount_ln'] < 1) | (df['sbfeaccountcount_ln'].isnull())) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & ((df['sbfehitindex_ln'] <= 1) | (df['sbfehitindex_ln'].isnull())) & (df['fico_score'].isnull())),
    (lambda df: (df['pg_required_c'] == False) & (df['sbfeaccountcount_ln'] >= 1) & (df['sbfehitindex_ln'] >= 2)),
    (lambda df: (df['pg_required_c'] == False) & (df['b2bcnt2y_ln'] >= 1) & (df['sbfehitindex_ln'] == 1)),
    (lambda df: (df['pg_required_c'] == False) & ((df['sbfeaccountcount_ln'] < 1) | (df['sbfeaccountcount_ln'].isnull())) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & ((df['sbfehitindex_ln'] < 1) | (df['sbfehitindex_ln'].isnull()))),
#     (lambda df: (df['pg_required_c'] == True) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].notnull())), # merged with segment 5
#     (lambda df: (df['pg_required_c'] == True) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].isnull())), # merged with segment 6, no hits
    (lambda df: (df['pg_required_c'] == False) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].notnull())),
    (lambda df: (df['pg_required_c'] == False) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].isnull()))
]

choices = [
    'pg_and_1_plus_sbfe_tradeline_and_fico_hit',
    'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit',
    'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit',
    'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit',
    'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit', 
    'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_no_hit', 
    'no_pg_and_1_plus_sbfe_tradeline',
    'no_pg_and_no_1_plus_sbfe_tradeline_and_1_plus_sba_tradeline',
    'no_pg_no_sbfe_no_sba_no_fico',
#     'pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_hit', ## contradiction between hitindex and b2bcnt
#     'pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_no_hit', ## contradiction between hitindex and b2bcnt
    'no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_hit', ## contradiction between hitindex and b2bcnt, how can have fico
    'no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_no_hit' ## contradiction between hitindex and b2bcnt
]
```


The creation of these segments gave us the opportunity to find new nuances in the data that were analyzed to make decisions. A set of new segments appeared that I called "false_1_plus_sba_tradeline" because they had a contradiction between the `sbfehitindex_ln` feature and the `b2bcnt2y_ln` feature.

In those cases, the hit index was equal to 1, meaning that the applicant was found only in the LexisNexis records and not in SBFE records, as this table explains:

https://drive.google.com/file/d/1HAKzRn9hlvfBVDSLHEwdCJjfqdyrMr8L/view

| Possible Value | Value Description |
|---|---|
| 0 | Business is not found in SBFE records or LexisNexis records |
| 1 | Business is found only in LexisNexis records |
| 2 | Business is found only in SBFE records |
| 3 | Business is found in SBFE records and LexisNexis records |

However, the column `b2bcnt2y_ln` has values 0, -99, or missing, indicating no tradelines in the records. This could be related to bankruptcies or other factors, meaning no SBFE tradeline and no b2b count. Then, that LexisNexis score essentially means that it is not based on tradeline data, so it's firmographic only.

Two of these segments were added to the current segments because "false SBA" indicates that there is no information about the tradelines (Commented code in Segmentation Logic). For the other segments, we decided to keep them but only use them as testing data to evaluate the potential effectiveness of a "voluntarily submitted PG" strategy.

#### Train: 2021

| id | Segment | Train | Train % Booked | Train Default rate in Booked applications | Train % Predicted as Booked | Train Default rate in Predicted Booked applications |
|---|---|---|---|---|---|---| 
| 1 | pg_and_1_plus_sbfe_tradeline_and_fico_hit | 2467 | 7.07% | 11.94% | 79.85% | 2.13% |
| 2 | pg_and_1_plus_sbfe_tradeline_and_fico_no_hit | 249 | 0.71% | 1.64% | 26.10% | 4.62% |
| 3 | pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit | 702 | 2.01% | 3.43% | 54.70% | 6.77% |
| 4 | pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit | 123 | 0.35% | 0.94% | 22.76% | 7.14% |
| 5 | pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit | 10853 | 31.12% | 27.84% | 64.56% | 3.28% |
| 6 | pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_no_hit | 1160 | 3.33% | 4.40% | 16.81% | 10.26% |
| 7 | no_pg_and_1_plus_sbfe_tradeline | 5337 | 15.30% | 20.42% | 85.35% | 3.10% |
| 8 | no_pg_and_no_1_plus_sbfe_tradeline_and_1_plus_sba_tradeline | 1682 | 4.82% | 4.86% | 76.40% | 10.58% |
| 9 | no_pg_no_sbfe_no_sba_no_fico | 1533 | 4.40% | 4.56% | 70.19% | 5.58% |
| 10 | no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_hit | 797 | 2.29% | 3.10% | 76.04% | 2.31% |
| 11 | no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_no_hit | 9969 | 28.59% | 16.86% | 81.06% | 7.05% |


#### Validation: Q4 2021

| id | Segment | Validation | Validation % Booked | Validation Default rate in Booked applications | Validation % Predicted as Booked | Validation Default rate in Predicted Booked applications |
|---|---|---|---|---|---|---| 
| 1 | pg_and_1_plus_sbfe_tradeline_and_fico_hit | 1344 | 21.49% | 79.85% | 2.13% | 85.33% |
| 2 | pg_and_1_plus_sbfe_tradeline_and_fico_no_hit | 185 | 4.59% | 26.10% | 4.62% | 78.31% |
| 3 | pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit | 386 | 3.84% | 54.70% | 6.77% | 34.62% |
| 4 | pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit | 106 | 1.02% | 22.76% | 7.14% | 0.81% |
| 5 | pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit | 3134 | 54.75% | 64.56% | 3.28% | 63.03% |
| 6 | pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_no_hit | 495 | 9.79% | 16.81% | 10.26% | 0.00% |
| 7 | no_pg_and_1_plus_sbfe_tradeline | 2299 | 2.49% | 85.35% | 3.10% | 84.90% |
| 8 | no_pg_and_no_1_plus_sbfe_tradeline_and_1_plus_sba_tradeline | 547 | 0.18% | 76.40% | 10.58% | 16.88% |
| 9 | no_pg_no_sbfe_no_sba_no_fico | 513 | 1.35% | 70.19% | 5.58% | 0.00% |
| 10 | no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_hit | 349 | 0.11% | 76.04% | 2.31% | 100.00% |
| 11 | no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_no_hit | 1898 | 0.40% | 81.06% | 7.05% | 0.00% |



#### Current: Since 02-16-2024 (Model implemented)

| id | Segment | Current | Current % Booked | Current Default rate in Booked applications | Current % Predicted as Booked | Current Default rate in Predicted Booked applications | Comments |
|---|---|---|---|---|---|---|---|
| 1 | pg_and_1_plus_sbfe_tradeline_and_fico_hit | 3256 | 73.51% | 7.29% | 76.56% | 4.18% |  |
| 2 | pg_and_1_plus_sbfe_tradeline_and_fico_no_hit | 695 | 19.46% | 0.00% | 71.35% | 0.00% |  |
| 3 | pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit | 582 | 52.85% | 15.69% | 38.08% | 8.84% |  |
| 4 | pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit | 154 | 9.43% | 30.00% | 0.94% | 0.00% |  |
| 5 | pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit | 8295 | 63.94% | 9.93% | 64.77% | 7.34% |  |
| 6 | pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_no_hit | 1483 | 13.54% | 10.45% | 0.00% | 0.00% |  |
| 7 | no_pg_and_1_plus_sbfe_tradeline | 377 | 80.03% | 9.62% | 82.51% | 6.48% |  |
| 8 | no_pg_and_no_1_plus_sbfe_tradeline_and_1_plus_sba_tradeline | 27 | 60.51% | 23.26% | 17.00% | 8.60% |  |
| 9 | no_pg_no_sbfe_no_sba_no_fico | 204 | 61.79% | 11.36% | 0.00% | 0.00% | Keep these segments, but only use them as testing data to evaluate the potential effectiveness of a "voluntarily submitted PG" strategy. These segments historically had FICO scores available, even though PGs weren't requested. |
| 10 | no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_hit | 17 | 67.91% | 8.86% | 100.00% | 6.02% | Keep these segments, but only use them as testing data to evaluate the potential effectiveness of a "voluntarily submitted PG" strategy. These segments historically had FICO scores available, even though PGs weren't requested. |
| 11 | no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_no_hit | 61 | 69.07% | 16.70% | 0.00% | 0.00% | Keep these segments, but only use them as testing data to evaluate the potential effectiveness of a "voluntarily submitted PG" strategy. These segments historically had FICO scores available, even though PGs weren't requested. |

### Validation of Segmentation

To ensure the effectiveness of our segmentation strategy, we performed the following checks:

* **Completeness:** We verified that all defined categories are populated and no application falls outside the defined scope.
* **Non-Empty Categories:** We confirmed that none of the categories are empty, ensuring all segments capture a portion of the applicant pool.

### Analysis of Segment Distribution

We analyzed the distribution of applications across different segments in both the test and scoring datasets. The results are summarized below:

**Test Data:**

| risk_grade_path | count |
|---|---|
| no_pg_1_plus_sbfe_trade_line | 2299 |
| pg_1_plus_sbfe_trade_line_fico_hit | 1344 |
| pg_just_fico_hit | 751 |
| no_pg_1_plus_sba_trade_line | 547 |
| no_pg_no_sbfe_no_sba_no_fico | 513 |
| pg_no_sbfe_1_plus_sba_trade_line_fico_hit | 386 |
| pg_1_plus_sbfe_trade_line_fico_no_hit | 185 |
| pg_no_sbfe_no_sba_no_fico | 170 |
| pg_no_sbfe_1_plus_sba_trade_line_fico_no_hit | 106 |
| **Total** | **6301** | 


**Scoring Data:**

| risk_grade_path | count |
|---|---|
| pg_1_plus_sbfe_trade_line_fico_hit | 5470 |
| no_pg_1_plus_sbfe_trade_line | 3099 |
| pg_just_fico_hit | 2144 |
| pg_no_sbfe_1_plus_sba_trade_line_fico_hit | 1102 |
| pg_1_plus_sbfe_trade_line_fico_no_hit | 1056 |
| no_pg_no_sbfe_no_sba_no_fico | 670 |
| no_pg_1_plus_sba_trade_line | 578 |
| pg_no_sbfe_no_sba_no_fico | 505 |
| pg_no_sbfe_1_plus_sba_trade_line_fico_no_hit | 315 |
| **Total** | **14939** |

### Observations:

* The segment distribution varies between the training and scoring datasets. This highlights the importance of  monitoring segment representation across different datasets.
* Notably, the proportion of "fico no hit" applications is larger in the training data compared to the scoring data.

### Conclusion:

The defined segments effectively capture the different applicant profiles and are well-populated across both test and scoring datasets. The observed shift in segment distribution between datasets underscores the dynamic nature of applicant characteristics and the need for continuous monitoring and model retraining. 

## 11/20/2024 10:40 AM

### Score Normalization

To prevent scores from being overweighted due to variance, we normalized them before training the model. This process ensures that all scores contribute equally to the model's predictions.

### Independent Logistic Regression Models

We trained independent logistic regression models for each of the 9 segments. This approach allows for more accurate predictions by tailoring the model to the specific characteristics of each segment.

### Normalization Scores and Estimated Parameters

The following table shows the normalization scores and estimated parameters for each segment:

**Normalization Scores:**

| Segment | Score Type | Standard Deviation | Mean |
|---|---|---|---|
| pg_sbfe_ln_and_fico | ln_score | 54.74 | 691.59 |
| pg_sbfe_ln_and_fico | fico_score | 63.85 | 710.16 |
| pg_sbfe_ln_only | ln_score | 66.71 | 683.92 |
| pg_sba_ln_and_fico | ln_score | 45.73 | 685.57 |
| pg_sba_ln_and_fico | fico_score | 72.54 | 647.54 |
| pg_sba_ln_only | ln_score | 49.15 | 674.62 |
| pg_fico_only | fico_score | 74.21 | 678.37 |
| no_pg_sbfe_ln_only | ln_score | 62.04 | 695.62 |
| no_pg_sba_ln_only | ln_score | 50.58 | 682.01 | 

 **Estimated Parameters:**

| segment | intercept | ln_score_z | fico_score_z |
|---|---|---|---|
| pg_sbfe_ln_and_fico | -4.256445 | -0.817988 | -0.972345 |
| pg_sbfe_ln_only | -4.081015 | -1.906164 | NaN |
| pg_sba_ln_and_fico | -2.292085 | -0.465804 | -1.041547 |
| pg_sba_ln_only | -2.584253 | 0.076314 | NaN |
| pg_fico_only | -3.163526 | NaN | -1.245019 |
| no_pg_sbfe_ln_only | -3.862532 | -0.997345 | NaN |
| no_pg_sba_ln_only | -2.237549 | -0.656777 | NaN |


### Model Performance

We evaluated the model's performance on scoring data from 2022 to the present. The table below shows the distribution of booked and non-booked applications, both actual and predicted, for each segment:

|segment_name|total_rows|num_booked|num_non_booked|predicted_booked|predicted_declined|
|---|---|---|---|---|---|
|pg_sbfe_ln_and_fico|5470|3776|1694|4291|1179|
|pg_sbfe_ln_only|1056|521|535|844|212|
|pg_sba_ln_and_fico|1102|369|733|364|738|
|pg_sba_ln_only|315|39|276|2|313|
|pg_fico_only|2144|1120|1024|1344|800|
|pg_no_hits|505|161|344|0|505|
|no_pg_sbfe_ln_only|3099|2479|620|2635|464|
|no_pg_sba_ln_only|578|323|255|127|451|
|no_pg_no_hits|670|409|261|0|670|

- The model shows a greater inclination towards SBFE tradelines.
- Conversely, the model appears to be more conservative in segments with SBA tradelines or no tradelines at all.


## 11/20/2024 4:50 PM

## Model Retraining and Analysis

This section provides an overview of the model retraining subsequent analysis performed.

**Key Steps:**

1. **Risk Grade Matrix Generation:**  After retraining, a risk grade matrix was generated for each segment. This matrix provides a detailed breakdown of the risk levels associated with each segment.

3. **Swap Test Generation:** A swap test was conducted to evaluate the performance of the retrained model. 

**Evaluation Resources:**

* **Risk Grade Matrix and Swap Test Results:**  https://docs.google.com/spreadsheets/d/1c-7zgPlJoHckIRwCkgjgJDxCX1vSHVGDs3d4UPCnRRM/edit?usp=sharing
* **Updated Presentation Deck:** https://docs.google.com/presentation/d/1ePdwrILsn0MJBZEAdKstYTfU5C5lgHVK4PGMPy3aiPA/edit?usp=sharing
