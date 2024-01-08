
# Capstone Project

## Table of content
* [Overview](#overview)
* [Project Set Up and Installation](#project-set-up-and-installation)
* [Dataset](#dataset)
* [Automated ML](#automated-ml)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Screen Recording](#screen-recording)
* [References](#references)

## Overview
The dataset utilized in this project was sourced from the National Institute of Diabetes and Digestive and Kidney Diseases. The goal is to forecast whether a patient has diabetes based on specific diagnostic measurements. Two models were developed for this project: one using AutoML and another employing hyperparameters tuned through the HyperDrive model with a LogisticRegression classifier. Subsequently, we assessed the performance of both models, deploying the one with the superior performance and integrating it for consumption.

The AutoML experiment has a 78.39% accuracy while the HyperDrive experiment gave a 77.08%. Hence the AutoML model was registered as the best model and deployed as a web service. 

## Project Set Up and Installation

### AutoML Experiment in 'automl.ipynb' Notebook:

1. **Upload Notebooks:** Begin by uploading the necessary notebooks to the AzureML workspace.

2. **Compute Instance:** Create a new compute instance

3. **Compute Cluster:** Set up a CPU compute cluster for model training.

4. **Dataset Registration:** Register the Kaggle dataset as "diabetes_data_set" in the workspace and retrieve it later in the ML experiment.

5. **Experiment Execution:**
   - Load the workspace, dataset, and compute cluster.
   - Create a new experiment named 'automl-exp.'
   - Configure AutoML settings and information.
   - Submit the experiment to train the model.
   - Utilize the RunDetails widget to display experiment details, including accuracy rates of the runs.
   - Retrieve and register the best model in the workspace.
   - Deploy the best model as a web service using Inference & deployment configuration settings.
   - Test the endpoint by sending a JSON payload and receiving a response.
   - Enable application insights and service logs.

### HyperDrive Experiment in 'hyperparameter_tuning.ipynb' Notebook:

1. **Load Dependencies:** Load the necessary components, including the workspace, dataset, and compute cluster.

2. **Experiment Setup:**
   - Create a new experiment named 'hyperdrive_exp.'
   - Define an early termination policy.
   - Utilize Random Parameter Sampling for hyperparameter tuning.
   - Configure experiment settings.

3. **Training Script:** Develop the 'train.py' script for training the model.

4. **Experiment Execution:**
   - Submit the HyperDrive experiment.
   - Use the RunDetails widget to visualize experiment details, such as accuracy rates of the runs.

## Dataset

The dataset comprises records of females aged 21 and older from Pima Indian heritage, totaling 768 entries. The primary goal is to predict the presence or absence of diabetes in patients based on specific diagnostic measurements, with features including the number of pregnancies, plasma glucose concentration, diastolic blood pressure, skinfold thickness, serum insulin levels, body mass index, diabetes pedigree function, age, and the target column "Outcome" indicating diabetes (1) or not (0). The dataset, available at https://www.kaggle.com/mathchi/diabetes-data-set, was downloaded from Kaggle, uploaded, and registered in the workspace using the 'upload from local file' option in the ML Studio GUI Datasets tab. It is registered as 'diabetes_data_set' and accessed through the 'Dataset_get_by_name(ws, dataset_name)' command in both notebooks, facilitating consumption via the Python SDK. The task involves predicting the "Outcome" column based on the given input features.

![](screenshots/dataset.png)

## Automated ML
In this experiment, the AutoML settings and configurations have been carefully selected to optimize the model training process. The **experiment_timeout_minutes** is set to 30 minutes to prevent unnecessary resource consumption. **max_concurrent_iterations** is configured to 30 to enhance computational efficiency by allowing a maximum of 30 iterations to run concurrently. The **primary_metric** is chosen as 'accuracy,' a fitting metric for classification problems and the key evaluation criterion for model performance. **n_cross_validations** is set to 5, ensuring the dataset is divided into five equal sets during cross-validation for robust evaluation. The number of iterations for the experiment is set to 20 with **iterations**, a reasonable value expected to yield meaningful insights. The **compute_target** is aligned with the project cluster for efficient execution. With the task of predicting whether a patient has diabetes or not, **task** is set to 'classification.' **training_data** refers to the loaded dataset, forming the basis for model training. **label_column_name** is set to 'Outcome,' representing the target column with values 0 or 1. Enabling **early_stopping** terminates the experiment if the accuracy score shows no improvement over time, optimizing resource utilization. **Featurization** is set to 'auto,' indicating an automated preprocessing step for numerical columns. Any errors during the process are logged into 'automl_errors.log' with **debug_log**, facilitating the identification and resolution of issues encountered during AutoML execution.

### Results
The best model has resulted from the AutoML experiment from VotingEnsemble model. The Voting Ensemble model takes a majority vote of several algorithms which makes it surpass individual algorithms and minimize the bias. The best model has a 78.39% accuracy rate. 

![](screenshots/automl_best_model.png)

#### `RunDetails` widget of best model screenshot 
![](screenshots/auto_ml_widget.png)

#### Best model run id screenshot
We can get the RunID from the provided notebook or simply look in the model section, after we registered our model.
![](screenshots/model_run_ids.png)

### Closer Look at the best model
The voting ensemble consists of 9 models with different preprocessing steps (scalers and normalizers) followed by various models. In the screenshot we see Random Forests,, Logistic Regression, XGB Models, LightGBM and Extreme Random Forests. All their classifications are weighted and combined for the final prediction. We see that the first ensemble part consisting of the ´StandardScalerWrapper´ and ´RandomForest´ has a model weight of roughly 0.077.
![](screenshots/automl_best_model_details.png)

The model consists
### How to improve the project in the future:
- Interchange n_cross_validations value between (2 till 7) and see if the prediction accuracy improved by tuning this parameter. 
- Increase the number of iterations this could lead to more improved results by testing more machine learning algorithms and run the experiment using them. 

## Hyperparameter Tuning

For the HyperDrive experiment, I opted for a LogisticRegression classifier model, given the nature of our target as a classification problem with binary outcomes (0 or 1). Logistic regression proves effective in such scenarios, utilizing the probability of a particular class occurrence. The model employs a logistic model function, constituting a binary regression form, and is trained using the provided 'train.py' script.

In configuring the HyperDrive experiment, I utilized the RandomParameterSampling for tuning hyperparameters, specifically '--C' (inverse of regularization) and '--max_iter' (maximum number of iterations). Discrete values, set as 'choice,' were assigned for both tuned parameters: '--C' with choices (0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100, 1000) and '--max_iter' with choices (10, 50). RandomParameterSampling was chosen for its swift performance, straightforward approach, and ability to provide a random, unbiased search in the overall parameter space. This approach is resource-efficient and supports the early termination policy for low-performance runs, saving computational resources. While Grid Sampling could offer an exhaustive search, the consideration of resource budget led to the selection of RandomParameterSampling.

To enhance resource utilization further, an early termination policy was incorporated into the experiment submission. BanditPolicy was employed with parameters evaluation_interval=2 and slack_factor=0.1, serving as an early stopping policy to automatically terminate underperforming and delayed runs. The Bandit Policy concludes runs if the primary metric falls outside the specified slack factor compared to the highest-performing run, contributing to improved computational efficiency.

### Results
The best performing model has a 77.08% accuracy rate with --C = 1 and --max_iter = 50. 

#### `RunDetails` widget screenshot of the best hyperdrive model
![](screenshots/hd_run_widget.png)
![](screenshots/hd_best_model.png)
#### Best model run id screenshot
As before, we take a look at the registered models to get the `RunID`
![](screenshots/model_run_ids.png)

### How to improve the project in the future:
In enhancing the experiment, consider delving into the impact of regularization by experimenting with a uniform range between 1 and 5 for the '--C' parameter. This exploration aims to unveil insights into the overall enhancement of performance and the model's ability to generalize. Additionally, broaden the scope for the '--max_iter' parameter, encompassing values such as 100 and 150, to evaluate how fine-tuning the number of iterations influences the model's performance.

To diversify the experimentation approach, contemplate introducing XGBoost and LightGBM models. Explore an extended range of hyperparameter options for these models to refine their performance, providing a more in-depth understanding of their behavior across diverse configurations.

In terms of early termination policies, experiment with both Median stopping and Truncation selection. Median stopping concludes runs based on the running averages of primary metrics, calculating averages across all training runs and removing the less-performing ones. This strategic approach aids in improving efficiency by concentrating resources on more promising configurations. Assess the efficacy of these policies in enhancing the overall efficiency and effectiveness of the experiment.

## Model Deployment
 The AutoML model exceeded the HyperDrive performance, hence we deploy it as a web service. The application insights was enabled.

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
![](screenshots/auto_ml_deploy_succeeded.png)

The deployment finished successfully and we see the endpoint is healthy.

#### Test the model
In order to test the model we copy the code from the ´Consume´ of a deployed model. There we choose the python code and get everything we need to test our model. 

![](screenshots/deploy_snippet.png)

Instead of using just zeros for the data to pass, we adapt the data parameter. The test data payload was passed to an instance of the endpoint model named "service". The prediction result was [1, 0] which indicates that only the first patient has diabetes. 

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
![](screenshots/automl_model_test.png)
## Screen Recording
![Link to screencast](https://www.youtube.com/watch?v=KkD5T8_6C68)

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
