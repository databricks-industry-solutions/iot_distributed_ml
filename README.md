<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-14.3ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/14.3lts-ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)

## Challenges Addressed
Today, field maintenance is often reactive, rather than proactive, which can lead to costly downtime and repairs. However, with Databricks businesses can implement predictive maintenance strategies that allow them to identify and address potential issues before they become customer facing problems. Databricks provides end-to-end machine learning solutions including tools for data preparation, model training, and root cause analysis reporting. 

Scaling existing codebases and skill sets is a key theme when it comes to using Databricks for data and AI workloads, particularly given the large data volumes that are common in IOT and anomaly detection use cases. For instance, a business may be experiencing an increase in engine defect rates without a clear reason, and they may already have a team of data scientists who are skilled in using Pandas for data manipulation and analysis on small subsets of their data - for example, analyzing particularly notable trips one at a time. By using Databricks, these teams can easily apply their existing Pandas code to their entire large-scale IOT dataset, without having to learn a completely new set of tools and technologies to deploy and maintain the solution. Additionally, ML experimentation is often done in silos, with data scientists working locally and manually on their own machines on different copies of data. This can lead to a lack of reproducibility and collaboration, making it difficult to run ML efforts across an organization. Databricks addresses this challenge by enabling MLflow, an open-source tool for unified machine learning model experimentation, registry, and deployment. With MLflow, data scientists can collaboratively track and reproduce their experiments, as well as deploy their models into production.

## Reference Architecture
<img src='https://raw.githubusercontent.com/databricks-industry-solutions/iot_distributed_ml/master/images/reference_architecture.png' width=800>

## Authors
josh.melton@databricks.com

## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 

## License

&copy; 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].
