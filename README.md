<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px"> 

[![DBR](https://img.shields.io/badge/DBR-15.4_ML_LTS-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/15.4lts-ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-AZURE/AWS/GCP-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)

## Business Problem

The purpose of this solution accelerator is to demonstrate how generative AI can be used to address several common product metadata challenges observed during the product on-boarding process.  During product on-boarding, suppliers register new products with their retail partners.  Inconsistencies, missing details, abbreviations, misspellings, etc. often require manual intervention on behalf of the retailer before the product can be accepted into the retailer's systems, slowing time to market. Using generative AI, many of these challenges can be addressed in an automated manner.

## Authors

<ali.sezer@databricks.com>

<bryan.smith@databricks.com>

## Project support

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the [Databricks License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.

## License

&copy; 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| [transformers](https://huggingface.co/docs/transformers/en/index) | Hugging Face Transformers | Apache 2.0 | [PyPI](https://pypi.org/project/transformers/) |
| [vllm](https://docs.vllm.ai/en/latest/) | Fast and efficient inference for LLMs | Apache 2.0 | [PyPI](https://pypi.org/project/vllm/) |
| [Amazon-Berkeley Objects (ABO) Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) | A CC BY 4.0-licensed dataset of Amazon products with metadata, catalog images, and 3D models | Creative Commons Attribution 4.0 International Public License (CC BY 4.0) | [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)

