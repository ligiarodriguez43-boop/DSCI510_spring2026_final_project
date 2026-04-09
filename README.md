Introduction: 

Colon Cancer Survival Prediction through Machine Learning 
 
Colon cancer was once considered a disease mainly affecting middle-aged or older adults, but recent studies have shown a rise in colon cancer among younger people. To address this change, building a machine learning (ML) prediction model can be helpful to estimate survival rates across different age groups. The model would use datasets containing patient age, global incidence colon cancer cases, cancer stage, risk factors, gender, and survival outcomes. After cleaning and organizing the data, these factors would train ML models such as logistic regression or decision trees to predict survival. The models would then be evaluated to assess how accurately they predict survival rates across age groups, genders, and countries and how effective are clinical trials. By comparing results between younger and older patients, the model could uncover patterns in survival outcomes. 

Data Sources: 
Data Source #	Name/ Short Description 	Source URL	Type:
- API
- Web page
- File	List of Fields	Format: 
- json
- xml
- csv
- sql
- other	Have tried to access/collect data with python? yes/no	Estimated data size, number of data points you plan to use
1	NIH Colorectal Cancer Clinical Trials
	"https://clinicaltrialsapi.cancer.gov/api/v2"
	API	Healthcare 	json	No (Not Yet)	300 or more 
2	Colorectal Cancer Global Dataset & Predictions – contains global data on colon cancer cases	https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions/discussion	File	Healthcare	CSV	No (Not Yet)  	300 or more 
3	Colorectal Cancer Dietary and Lifestyle Dataset – colon cancer risk factors	https://www.kaggle.com/datasets/ziya07/colorectal-cancer-dietary-and-lifestyle-dataset	File 	Healthcare 	CSV	No (Not Yet) 	300 or more
4	CDC Wonder 	https://wonder.cdc.gov/wonder/help/wonder-api.html	File 	Healthcare	XML	No (Not Yet)	300 or more

