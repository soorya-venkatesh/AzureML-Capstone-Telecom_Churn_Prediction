
# Telecom Customer Churn Prediction

The goal of this project is to predict the churn of customers belonging to a Telecom company. AzureML's machine functionalities such as Hyperdrive and AutoML are extensively going to be used to train a model and the best model will be deployed as a webservice in Azure and the endpoint will be tested out. 

## Project Set Up and Installation
Make sure you have a Azure Machine learning account in order to reproduce the results.
- **Step 1**: Have a look at the Telecom_churn_EDA.ipynb for a EDA on the data
- **Step 2**: Use the hyperparameter_tuning.ipynb to run and run the cells in order to train the random forest model and tune its parameters using hyperdrive. The train.py contains the entry script which be utilized in the hyperdrive itself.
- **Step 3**: In order to deploy the model, you will have to register the model.joblib(provided) in AzureML. Once it is registered you can deploy it giving the conda dependency file (conda_dependencies.yml) and the entry script for inference(score.py)
- **Step 4**: In order to train the AutoML model you can use the automl.ipynb and execute the cells in order     


## Dataset

### Overview
The data is about a telco company that provided home phone and Internet services to 7043 customers in California in Q3. It contains 7043 observations with 33 variables. The data is got from kaggle and can found through this [link](https://www.kaggle.com/yeanzc/telco-customer-churn-ibm-dataset)

### KPI Used:
Macro averaged AUC is used to compare different model because the problem is a imbalanced classification problem and AUC score accounts for the class imbalance.  

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
The AutoML model was trained 3 fold cross validation, was trained as a classification task for a maximum of 15 mins. Early stpping was enabled. A maximum of 6 concurent runs could happen at a time. The cluster used was a STANDARD_D13_V2 with 8 cores. The full list of settings is given below
```
automl_settings = {
       "n_cross_validations": 3,
       "primary_metric": 'AUC_weighted',
       "enable_early_stopping": True,
       "experiment_timeout_minutes": 15,
       "max_concurrent_iterations": 6
   }

# automl config 
automl_config = AutoMLConfig(task = "classification",
                               compute_target = cpu_cluster,
                               training_data = ds,
                               label_column_name = "Churn Value",
                               **automl_settings
                            )
```

### Results
As stated previously a AutoML model was trained to classify the customer churn. The below image shows the completed experiment for the AutoML run
![experiment_completed](https://user-images.githubusercontent.com/63422900/154573478-8c58a075-1c2d-4fcb-9a60-d5fb2c003d56.PNG)

The best model got was a Voting Ensemble model as shown below. The best macro averaged AUC for this model was 0.864.  
![best_model](https://user-images.githubusercontent.com/63422900/154573727-26b951e4-214b-4951-b2e4-11359c404344.PNG)

The various list of models used in the AutoML experiment is shown below.
![model-list1](https://user-images.githubusercontent.com/63422900/154574607-a6433eb8-d6f0-4a74-aa12-509a6a2ea22f.PNG)
![model-list2](https://user-images.githubusercontent.com/63422900/154574626-ed0f75bd-bfbe-45e3-8f19-208542466a04.PNG)

The run details widget for the AutoML run is shown below
![run_details1](https://user-images.githubusercontent.com/63422900/154574938-9750c632-e436-462c-a3ad-c8f95473ef3b.PNG)
![run_details2](https://user-images.githubusercontent.com/63422900/154574948-de3bde69-f16f-441f-b59a-9e17e71ed98c.PNG)


## Hyperparameter Tuning
A random forest model was trained and its hyper paramteres were tuned using hyperdrive. The parameters used to tune were 
- max_depth: maximum depth per tree
- n_estimators: Total decision tress that can be used
- min_samples_split: The minimum number of samples required to split an internal node
- min_samples_leaf: The minimum number of samples required to be at a leaf node

### Parameter Sampling Stratergy:
Random Search was used to find the best value of "C" and "max_iter". The search space for "C" is [0.0001,1000] and search space for "max_iter" was {75,125}.
#### Benifits of Parameter Sampling:
In any machine learning problem in order to find the optimal value(s) of weights , a search stratergy is used. Some of them are Random search (picking hyperparameter combinations at random and choosing the one with least loss or best metric ) , grid search (iteratively going through all possibilities of hyperparameter combinations and choosing the one with the least loss or best metric) , Bayesian search (using Bayesian stratergies to search for the best hyperparameter combination) etc.  

### Early Stopping Policy:
A bandit policy of slack factor=0.17 and evaluation_interval=2 is being used. A slack factor=0.17 will terminate the training when if the metric at the current iteration is 17% less than the best performing metric. evaluation_interval=2 means that this evaluation will happen once every 2 iterations.

More details about the config are shown below
```
ps = RandomParameterSampling( {
    'max_depth': choice(range (2, 5, 1)),
    'n_estimators': choice(range(50, 560, 100)),
    'min_samples_split': choice(2,3,5),
    'min_samples_leaf':choice(1,3,5)   
    }
)

policy = BanditPolicy(slack_factor = 0.17, evaluation_interval=2, delay_evaluation=5)

est = ScriptRunConfig(source_directory='.',
                            script='train.py',
                            compute_target=cpu_cluster,
                            environment=env)

hyperdrive_config = HyperDriveConfig(run_config=est,
                                     hyperparameter_sampling=ps,
                                     policy=policy,
                                     primary_metric_name='AUC-ROC',
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_concurrent_runs=8,
                                     max_total_runs=15)
```

### Results
The below image shows the completed hyperdrive experiment run
![experiment_completed](https://user-images.githubusercontent.com/63422900/154577041-00a85c9c-dadd-4788-b331-776e4f1f3f7c.PNG)

The best run got was a AUC score of 0.867 with the following parameters as shown below.
- max_depth: 3
- n_estimators: 450
- min_samples_split: 3
- min_samples_leaf: 1
![best_child_run](https://user-images.githubusercontent.com/63422900/154577331-5543eb0e-3ff0-4213-91b7-8f9b379108fc.PNG)
The AUC score over different runs is given below.
![auc_score](https://user-images.githubusercontent.com/63422900/154577462-4ca40aa5-2c0d-4eb8-b7d1-8ba5f2e0a90d.PNG)
The various hyperparameters for different runs is graphically represented below
![parameters_graph](https://user-images.githubusercontent.com/63422900/154577504-71ef746b-a39f-4e5b-bce2-7e4e54b09893.PNG)
The run widgets hyperdrive for the hyper drive experiment is given below
![run_widgets_hyperdrive](https://user-images.githubusercontent.com/63422900/154577625-63c031f7-cbcd-4293-a900-f89e3d2402eb.PNG)


## Model Deployment
The best AUC got from the random forest with hyperdrive experiment was 0.87 and the best AUC got from the AutoML experiment was 0.862. Thererfore the random forest model was chosen for deployment. The deployment was done using the Python SDK for AzureML.
From the below image it can be seen that the deployment has happened successfully and it is in healthy state.
![healthy_state](https://user-images.githubusercontent.com/63422900/154579941-39dc7616-6940-4412-9516-848ed81e81b3.PNG)

Application insights was enabled which means the server side logs such as request over time, failed requests, active period can be got for the deployed model at any point in time.
![application_insights](https://user-images.githubusercontent.com/63422900/154580265-4a16f038-3be7-45f3-9aca-a5400559ea76.PNG)

#### Quering the deployment to get output 
Here 3 input points are taken from x_test and it is combined along with the header in the form of a JSON and a post request  is given to the deployment URL. 
```
req = x_test[:3].values.tolist()
req = [req[0][:],req[1][:],req[2][:]]

scoring_uri='http://ef864eb6-2c46-4540-8f46-7e6cca62315b.southcentralus.azurecontainer.io/score'
data = json.dumps({"data":req})
headers = {'Content-Type':'application/json'}

response = requests.post(scoring_uri,data,headers=headers)
print(response.text)
```
For the request we get a response of [0,0,0] which means for the 3 customers input points which was given in the POST request , the model predicted as not churned for these 3 points.

*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording

**Part 1:**


https://user-images.githubusercontent.com/63422900/154582558-621f0e53-c0f8-4cfa-ae86-bef5e370812e.mp4

**Part 2:**


https://user-images.githubusercontent.com/63422900/154584601-d224cb99-e333-4e85-8b7d-08fa05953ffd.mp4





## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
