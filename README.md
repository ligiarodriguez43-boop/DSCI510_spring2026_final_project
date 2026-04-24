Introduction: 

Colon Cancer Survival Prediction through Machine Learning Models 
 
Colon cancer was once considered a disease mainly affecting middle-aged or older adults, but recent studies have shown a rise in colon cancer among younger people. To address this change, building various machine learning (ML) prediction models can be helpful to estimate mortality rates. The model would use datasets containing patient age, global colon cancer data, cancer stage data, risk factors, gender, and survival outcomes. After cleaning and organizing the data, these factors would train ML models such as logistic regression, decision trees, and gradient boosting to predict mortality. The models would then be evaluated to assess how accurately they predict mortality rates across age groups, genders, and countries and how effective are colon cancer clinical trials.

Data Sources: 
| Source # | Name/Source | URL              | Type | List of Fields | Format | Estimated Data Size, number of data points you plan to use |
| :--------:| :-----------: | :--------------: | :-----: | :--------------: | :------: |  :----------------------------------------------------------: |
| 1        | NIH Colorectal Cancer Clinical Trials | https://clinicaltrialsapi.cancer.gov/api/v2 | API | Healthcare | JSON | 300 or more |
| 2        | Colorectal Cancer Global Dataset  | https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions/discussion | File | Healthcare | CSV | 167,497 | 
| 3        | Colorectal Cancer Dietary and Lifestyle Dataset – colon cancer risk factors | https://www.kaggle.com/datasets/ziya07/colorectal-cancer-dietary-and-lifestyle-dataset | File | Healthcare | CSV | 1,000 | 
| 4        | CDC Wonder   | https://wonder.cdc.gov/wonder/help/wonder-api.html | Web | Healthcare | XML | 300 

