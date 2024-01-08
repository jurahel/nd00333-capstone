
# Capstone Project: Diabetes prediction

## Table of content
* [Overview](#overview)
* [Project Set Up and Installation](#project-set-up-and-installation)
* [Dataset](#dataset)
* [Automated ML](#automated-ml)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Screen Recording](#screen-recording)
* [References](#references)

## Overview
The used dataset originally has been taken from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the project is to predict if a patient has diabetes or not by evaluating certain diagnostic measurements. In this project, we have created two models: one using AutoML and one using hyperparameters tuned using HyperDrive model with LogisticRegression classifier. Then, we compared the performance of both the models and deploy the best performing model and consume it.

![](Screenshots/steps.png)
###### Source: (https://medium.com/microsoftazure/9-advanced-tips-for-production-machine-learning-6bbdebf49a6f)

The AutoML experiment has a 78.39% accuracy while the HyperDrive experiment gave a 74.4%. The AutoML model exceeded the HyperDrive performance by 3.99%, Hence was registered as the best model and deployed as a web service. 


## Project Set Up and Installation

To run the project, follow the steps below:

- Upload notebooks to the AzureML workspace.
- Create a new compute instance to run the notebook script, STANDARD_DS2_V12 was selected.
- Create a cpu compute cluster of type STANDARD_DS12_V2 with 4 max nodes, and low priority status to train the model.
- Registered Kaggle dataset to the workspace under the name "diabetes_data_set", and retrieved it later in the ML experiment.

- Run the AutoML experiment through 'automl.ipynb' notebook as follow: 
  - Load the ws, dataset, compute cluster.
  - Create a new experiment named 'automl-exp'.
  - Add the AutoML settings and configuration information, then submit the experiment to train the model.
  - Use the RunDetails widget to show experiment details such as the runs accuracy rate.
  - Retrive the best model and registerd it in the workspace.
  - Deploy the best model as a web service using Inference & deployment configuration settings. 
  - Test the endpoint by sending json payload and receive a response.
  - Enable the application insights and service logs.

- Run the HyperDrive experiment through 'hyperparameter_tuning.ipynb' notebook as follow: 
  - Load the ws, dataset, compute cluster.
  - Create a new experiment named 'hyperdrive_exp'.
  - Define early termination policy, Random Parameter Sampling  hyperparmenter and config settings. 
  - Create 'train.py' script to be used in training the model, then submit the experiment.
  - Use the RunDetails widget to show experiment details such as the runs accuracy rate.

## Dataset

### Overview
The dataset collects the records of females patients of age 21 and older from Pima Indian heritage. The dataset has a total of 768 entries. The objective is to predict if a patient has diabetes or not by evaluating certain diagnostic measurements.
https://www.kaggle.com/mathchi/diabetes-data-set

### Task
Predict the "Outcome" column based on the input features, either the patient has diabetes or not. 

The dataset has nine features as follow:
- Pregnancies: Number pregnancy times (int).
- Glucose: Plasma glucose concentration level (int). 
- BloodPressure: Diastolic blood pressure level in mm Hg(int).
- SkinThickness: skinfold thickness in mm(int).
- Insulin: two-hour serum insulin measured by mu U/ml(int).
- BMI: Body mass index(float).
- DiabetesPedigreeFunction: Diabetes pedigree function(float).
- Age: age in years 21 and above(int).
- Outcome: Target column 0 or 1, 0 = Not diabetes, 1 = diabetes(int).

### Access
The dataset was downloaded from kaggle then uploaded/registered in the workspace through 'upload from local file' option in the ML Studio GUI Datasets tab. The dataset was registered with the name 'diabetes_data_set' and could be accessed via 'Dataset_get_by_name(ws,dataset_name)' command. It was consumed by both notebooks using Python SDK.

![](Screenshots/registerd-dataset.png)

## Automated ML
Overview of the `automl` settings and configuration used for this experiment:

- "experiment_timeout_minutes": set to 30 minutes. The experiment will timeout after that period to avoid wasting resources.
- "max_concurrent_iterations": is set to 4. The max number of concurrent iterations to be run in parallel at the same time.
- "primary_metric" :  is set to 'accuracy', which is a sutible metric for classification problems. 
- "n_cross_validations": is set to 5, therefore the training and validation sets will be divided into five equal sets.
- "iterations": the number of iterations for the experiment is set to 20. It's a reasonable number and would provide the intendable result for the given dataset.
- compute_target: set to the project cluster to run the experiment.
- task: set to 'classification' since our target to predict whether the patient has diabetes or not.
- training_data: the loaded dataset for the project.
- label_column_name: set to the result/target colunm in the dataset 'Outcome' (0 or 1).
- enable_early_stopping: is enabled to terminate the experiment if the accuracy score is not showing improvement over time.
- featurization = is set to 'auto', it's an indicator of whether implementing a featurization step to preprocess/clean the dataset automatically or not. In our case, the preprocessing was applied for the numerical columns which normally involve treating missing values, cluster distance, the weight of evidence...etc.
- debug_log: errors will be logged into 'automl_errors.log'. 

### Results
The best model has resulted from the AutoML experiment from VotingEnsemble model. The Voting Ensemble model takes a majority vote of several algorithms which makes it surpass individual algorithms and minimize the bias. The best model has a 78.39% accuracy rate. 

![](Screenshots/automl-models.png)

#### `RunDetails` widget of best model screenshot 
![](Screenshots/automl_run_dtl_p1.png)

![](Screenshots/automl-matix.png)

#### Best model run id screenshot
![](Screenshots/automl-registered-model.png)

### How to improve the project in the future:
- Interchange n_cross_validations value between (2 till 7) and see if the prediction accuracy improved by tuning this parameter. 
- Increase the number of iterations this could lead to more improved results by testing more machine learning algorithms and run the experiment using them. 

## Hyperparameter Tuning

I have chosen a LogisticRegression classifier model to train the HyperDrive experiment. Since our target is to predict classification problem result either 0 or 1. The model uses the probability of a certain class to occur. Logistic regression uses a logistic model function which is a form of binary regression. The model is trained using 'train.py' script. 

The used HyperDrive parameters: 
- Parameter sampler 'RandomParameterSampling' holds the tuning hyperparameters (--C: Inverse of regularization, --max_iter: Maximum number of iterations) was passed to the HyperDriveConfig script.
  - Discrete values with 'choice' have been used for both tuned parameters '--C' : choice(0.001,0.01,0.1,1.0,10.0,50.0,100,1000),
        '--max_iter': choice(10,50). RandomParameterSampling has been selected due to its fast performance, simple approach, and would provide random unbiased search in the overall population. In addition, it gives satisfactory results and supports the early termination policy of low-performance runs which results in saving resources. Grid Sampling can be used for exhaustive search over the search space if the budget was not an issue.
- Early termination policy has been added to the script then experiment submission.
  - BanditPolicy has been used with the parameters evaluation_interval=2 and slack_factor=0.1 as an early stopping policy to improve the performance of the computational resources by automatically terminating poorly and delayed performing runs. Bandit Policy ends runs if the primary metric is not within the specified slack factor/amount when compared with the highest performing run.

### Results
The best performing model has a 74.4% accuracy rate with --C = 50 and --max_iter = 50. 

#### `RunDetails` widget screenshot of the best model
![](Screenshots/hd_run_dtl_p1.png)

![](Screenshots/hd_run_dtl_p2.png)

![](Screenshots/hd_run_dtl_p3.png)

![](Screenshots/hd_run_dtl_p4.png)

#### Best model run id screenshot
![](Screenshots/hd-best-run-id.png)


### How to improve the project in the future:
- Try a uniform range between 1 and 5 for regularisation (--C) to see the overall improvement in the performance and generalization capability.
- Increase the number of --max_iter to cover 100 and 150 and evaluate the impact of tuning the iterations on the model performance.
- Consider using XGBoost, and LightGBM models for the experiment and explore more Hyper Parameters options to tune.
- Try Median stopping, and Truncation selection early termination policies. Median stopping terminates runs based on the running averages of primary metrics. Thus, computing all training runs averages and eliminate the worse runs.

## Model Deployment
The AutoML experiment has a 78.39% accuracy while the HyperDrive experiment gave a 74.4%. The AutoML model exceeded the HyperDrive performance by 3.99%, Hence was registered as the best model and deployed as a web service. The application insights was enabled.

Also, we have created inference configuration and edited deploy configuration settings for the deployment. The inference configuration and settings explain the set up of the web service that will include the deployed model. Environment settings and scoring.py script file should be passed the InferenceConfig. The deployed model was configured in Azure Container Instance(ACI) with cpu_cores and memory_gb parameters initialized as 1. 

```
inference_config = InferenceConfig(entry_script='scoring.py',
                                   environment=environment)
service_name = 'automl-deploy-1'
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config,
                       overwrite=True
                      )
service.wait_for_deployment(show_output=True)
```

#### Best Model screenshot
![](Screenshots/automl-run-detl-p1.png)

![](Screenshots/automl-run-detl-p2.png)

![](Screenshots/deployed-model.png)

![](Screenshots/deployed-model-healthy.png)

![](Screenshots/deployed-enable-insights.png)

A two sets of test records were passed to the endpoint to predict the result and use the service. The test data payload was passed to an instance of the endpoint model named "service". The prediction result was [1, 0] which indicates that only the first patient has diabetes. 

```
data = [{
           "Pregnancies": 6, 
             "Glucose": 148, 
             "BloodPressure": 72, 
             "SkinThickness": 35, 
             "Insulin": 0, 
             "BMI": 33.5, 
             "DiabetesPedigreeFunction": 0.627, 
             "Age": 50
           },
          {
            "Pregnancies": 1, 
             "Glucose": 85, 
             "BloodPressure": 66, 
             "SkinThickness": 29, 
             "Insulin": 20, 
             "BMI": 26.5, 
             "DiabetesPedigreeFunction": 0.351, 
             "Age": 31
          },
      ]
  # test using service instance
  input_data = json.dumps({
  'data': data
   })

   output = service.run(input_data)
   print(output)
```
#### Endpoint Result
![](Screenshots/input-json-payload.png)

![](Screenshots/json-response.png)

The endpoint result can be run by 'endpoint.py' script. The script can be executed using 'python endpoint.py' command to receive the prediction. 
![](Screenshots/endpoint-py.png)

## Screen Recording
A link to a screen recording of the project is [Click Here](https://www.loom.com/share/431c6a9c7ef64fc5927f4b9248c9e6cb)
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

## References
- Udacity Nanodegree Content.
- https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-power-bi-custom-model
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-existing-model
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments
- https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-bring-data
- https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python#define-an-entry-script
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-enable-app-insights
- https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py
- https://en.wikipedia.org/wiki/Logistic_regression
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters
- https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines
- https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py
- https://medium.com/microsoftazure/9-advanced-tips-for-production-machine-learning-6bbdebf49a6f
