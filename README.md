
# Telecom Customer Churn Prediction

The goal of this project is to predict the churn of customers belonging to a Telecom company. AzureML's machine functionalities such as Hyperdrive and AutoML are extensively going to be used to train a model and the best model will be deployed as a webservice in Azure and the endpoint will be tested out. 

## Project Set Up and Installation
Make sure you have a Azure Machine learning account in order to reproduce the results.
Step 1: Use the hyperparameter_tuning.ipynb to run and run the cells in order to train the random forest model and tune its parameters using hyperdrive. The train.py contains the entry script which be utilized in the hyperdrive itself.
Step 2: In order to deploy the model, you will have to register the model.joblib(provided) in AzureML. Once it is registered you can deploy it giving the conda dependency file (conda_dependencies.yml) and the entry script for inference(score.py)
Step 3: In order to train the AutoML model you can use the automl.ipynb and execute the cells in order     


## Dataset

### Overview
The data is about a telco company that provided home phone and Internet services to 7043 customers in California in Q3. It contains 7043 observations with 33 variables. The data is got from kaggle and can found through this [link](https://www.kaggle.com/yeanzc/telco-customer-churn-ibm-dataset)

### Task
The task will be to accurately predict the customer churn for this data. First a random forest model is trained and its hyperparameters are tuned using hyperdrive. Next a AutoML model is trained for the same data. The best model is deployed as a web service in Azure and the deployed endpoint is tested. The following are the features which are utilized for trainning the model.(The information for the features is got from the kaggle page)
- City: The city of the customer’s primary residence.
- Gender: The customer’s gender: Male, Female
- Senior Citizen: Indicates if the customer is 65 or older: Yes, No
- Partner: Indicate if the customer has a partner: Yes, No
- Dependents: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
- Tenure Months: Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
- Phone Service: Indicates if the customer subscribes to home phone service with the company: Yes, No
- Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
- Internet Service: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
- Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
- Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
- Device Protection: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
- Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
- Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.
- Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.
- Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
- Paperless Billing: Indicates if the customer has chosen paperless billing: Yes, No
- Payment Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
- Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.
- Total Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.
- Churn Value: 1 = the customer left the company this quarter. 0 = the customer remained with the company. Directly related to Churn Label.
- CLTV: Customer Lifetime Value. A predicted CLTV is calculated using corporate formulas and existing data. The higher the value, the more valuable the customer. High value customers should be monitored for churn.
- Churn Reason: A customer’s specific reason for leaving the company. Directly related to Churn Category.

### Access
In order to access the data firstly the data has to be downloaded from Kaggle and uploaded to github. The the raw format link ("https://raw.githubusercontent.com/soorya-venkatesh/nd00333-capstone/master/starter_file/Telco_customer_churn.csv") can be used directly with the TabularDatasetFactory class in AzureML SDK.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
