## Milestone 1 Proposal

- Note this proposal was completed within the README file for Milestone 1 submission on January 18, 2020
    - Archived to this file January 24, 2020

### Section 1: Selected Data Set

Our selected data set summarizes student's achievements in secondary education for two Portuguese schools. The data was compiled from official school reports as well as questionaires provided to students. Attributes include the student grades, demographic, social and other school related features. Two datasets are provided regarding the performance in two distinct subjects: Mathematics (`mat`) and Portuguese language (`por`). 

Both datasets can be summarised by the following highlights:
- The key feature of interest is `G3` which represents the final subject grade score at the end of the academic year. The score ranges from 0 to 20. 
- Multi-categorical attributes such as family size (`famsize`), parent's education (`Fedu`/`Medu`) or weekly study time (`studytime`)
- Binary attributes such as home internet access (`internet`) or family educational support (`famsup`)
- Count attributes such as number of days of absence (`absences`) or student's age (`age`)

For additional information, refer to the following:
- The data set's [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance) summary page.
- [P. Cortez](http://www3.dsi.uminho.pt/pcortez/Home.html) and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7. 
    - The original research [paper](http://www3.dsi.uminho.pt/pcortez/student.pdf)

### Section 2: Research Question

__Main predictive question:__ 

For both subjects in Math and Portuguese, what are the top five features that most strongly predict high school student's performances (`G3 Score`)?

__Sub questions:__ 
 - Which machine learning model type (RandomForest, Linear Regression, XGBoost, etc) is able to predict most accurately for both subjects?
 - For the best ML model type, which features are the top 5 strongest predictors for each subject?
 - Additional insight: Are there common top predictors for the grade predictions of Math and Portuguese?

### Section 3: Analysis Plan of Action

<u>Project Steps:</u>  
Our research direction is to determine the top predictors for grade prediction of both subjects. The project will comprise of the following:
- __Data Split:__ Split both (Math and Portugese) datasets into two training and test sets (80% to 20% ratio). 
- __Models:__ Select ML model types with intepretable feature extraction.
- __EDA:__ For each training dataset, perform visual analysis with focus on exploring on highly correlated attributes.
- __Training:__ Perform cross-validation model training for different ML models using training datasets.
- __Testing:__ Perform testing for each ML model type with optimised hyper parameters. Obtain the ML model type that has the best average scoring for grade prediction of both subjects.
- __Analysis:__ For the best ML Model type, extract out the top 5 features from both trained models (Math and Portuguese). The scoring of feature importance may be different for different ML model types and packages. The envisioned deliverable for this analysis is a visual plot of the top 5 predictors for subject grades and compare them side by side. Their corresponding importance can be illustrated with a lollipop chart type.

<u>EDA:</u>  
An initial general heatmap of correlation strength between attributes will be used to identify relevant features for prediction of `g3`. Thereafter,  boxplots will be used to show the relationship across categorical/binary attributes with respect to the grades, and scatter plots will be used for count data attributes. Visualising the distributions will also help us spot any outliers in our dataset. As mentioned, EDA will only be done for the training dataset. 

Visual plots will be done with `Altair` and `Pandas Profiling`, and supplemented with summary statistics. All these are located in this [EDA script](./src/EDA.ipynb). Please note that to run our EDA Notebook, ensure that you have `Pandas Profiling` version 2.4.0 or later.

For a hands-on trial of our `Pandas Profiling`, please use this [Binder Link](https://mybinder.org/v2/gh/UBC-MDS/DSCI_522-Group-403-Student-Performance/bb8c88b74124c12bc4f029917791a4c0b2961792):  
- Link should direct you to an online Binder version of our project repo. 
- Thereafter, locate the "EDA.ipynb" under the main folder "src". 
- Note: Before running the notebook, please comment out the `alt.renderers.enable('notebook')` [last line] in the 1st cell.
- Run the relevant cells to view the `pandas profiling` output.