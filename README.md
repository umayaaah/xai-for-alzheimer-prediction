# Explainable AI for Alzheimer's Disease Classification
Using XAI methods on trained classifiers when predicting AD and MCI against normal controls.

#### Objective:
The repository is part of submission for the COMP3932 Final Year Project. The objectives are:
- Apply the fundamental concepts of machine learning using the ADNI (private) datasets
- Evaluate and train random forest models for Alzheimer's Disease (AD) and Mild Cognitive Impairement (MCI) classification problems
- Use interpretation frameworks to the models to generate plots for explanations
- Create notebooks that serve as computational records for 


The project uses jupyter notebooks to meet the objectives, in order to run certain experiments yourself, it is recommended to install the requirements using the command below inside a virtual environment.
```
pip install -r requirements.txt
```

[Identifying the problem and Data Sources]()
Identify the types of information contained in the ADNI dataset for the purpose of getting to know what data is available and  the data to get a good grasp of the data and think about how to handle the data in different ways.

[Exploratory Data Analysis]()
 Data exploration and visualization techniques using python libraries (Pandas, matplotlib, seaborn). Familiarity with the data is important which will provide useful knowledge for data pre-processing and training the models.

[AD Classification]()
This folder has the following 4 stages as notebooks to build a model that has high accuracy in performance for classifying AD with normal controls patients. The best model achieved an accuracy of 96.12%.
1. Training model
2. Evaluating model
3. LIME framework
4. SHAP framework

[MCI Classification]()
This folder has the following 4 stages as notebooks to build a model that has high accuracy in performance for classifying MCI with normal controls patients. The best model achieved an accuracy of 86.77%.
1. Training model
2. Evaluating model
3. LIME framework
4. SHAP framework