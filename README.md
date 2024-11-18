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

# 11/18/2024 9:34 AM

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