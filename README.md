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

# 11/18/2024 11:34 AM

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

