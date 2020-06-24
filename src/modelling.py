# author: Kenneth Foo
# date: 2020-01-23
#
# This script will perform preprocessing on both training and test data, and create various models to predict the grades of Portuguese subject.
# It will output the best hyperparameters for each model's cross validation, and score the predictions of the different models
# Models used in this script are Linear Regression with Lasso, Linear Regression with Ridge, Random Forest Regressor, XGBoost Regressor and Light GBM Regressor.
#
# Outputs (relative path from project repo):
# Cross-validation results  (csv_output_dir_path + "cv_results.csv")
# lmlasso_hyperparam        (csv_output_dir_path + "lmlasso_hyperparam.csv")
# lmridge_hyperparam        (csv_output_dir_path + "lmridge_hyperparam.csv")
# rf_hyperparam             (csv_output_dir_path + "rf_hyperparam.csv")
# xgb_hyperparam            (csv_output_dir_path + "xgb_hyperparam.csv")
# lgbm_hyperparam           (csv_output_dir_path + "lgbm_hyperparam.csv")
# test_rmse                 (csv_output_dir_path + "final_results.csv")
# feat_importances          (csv_output_dir_path + "feat_importance.csv")
# Plot of top 5 feat        (image_output_dir_path + "ranked_features.png")
###################################################################
'''This script will perform preprocessing on both training and test data, and create various models to predict the grades of Portuguese subject.
It will output the best hyperparameters for each model's cross validation, and score the predictions of the different models
Models used in this script are Linear Regression with Lasso, Linear Regression with Ridge, Random Forest Regressor, XGBoost Regressor and Light GBM Regressor.

Usage: modelling.py --train_data_file_path=<train_data_file_path> --test_data_file_path=<test_data_file_path> --csv_output_dir_path=<csv_output_dir_path> --image_output_dir_path=<image_output_dir_path>

Options:
--train_data_file_path=<train_data_file_path>  Path (including filename) to the training data csv file.
--test_data_file_path=<test_data_file_path>  Path (including filename) to the test data csv file.
--csv_output_dir_path=<csv_output_dir_path>  Path (excluding any filenames) to the output csv directory. Must end with "/".
--image_output_dir_path=<image_output_dir_path>  Path (excluding any filenames) to the output image directory. Must end with "/".
'''

# Typical packages
import pytest
from docopt import docopt
import numpy as np
import pandas as pd

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Cross validation
from sklearn.model_selection import cross_validate

# Bayes opt
from bayes_opt import BayesianOptimization

# Linear Regression with Lasso
from sklearn.linear_model import Lasso
# Linear Regression with L2
from sklearn.linear_model import Ridge
# Random Forest
from sklearn.ensemble import RandomForestRegressor
# XGBoost
import xgboost as xgb
# LightGBM
import lightgbm as lgbm

# Scoring
from sklearn.metrics import mean_squared_error

# Plotting
import altair as alt

################################################################
opt = docopt(__doc__)

def main(train_data_file_path, test_data_file_path, csv_output_dir_path, image_output_dir_path):

    """ 
    This function performs the preprocessing and predictive modelling, and outputs various csv files on crossvalidation scores, 
    model hyperparameters, and plot of top 5 predictive features based on best model.
    
    Parameters
    ----------
    train_data_file_path: str
        A string that provides a FILE path (including filename) in which the training data is stored.
        Cannot be null, otherwise an error will be thrown.
    
    test_data_file_path: str
        A str that provides a FILE path (including filename) in which the test data is stored.
        Cannot be null, otherwise an error will be thrown.

    csv_output_dir_path: str
        A string that provides the DIRECTORY path (including "/" character at the end) to store csv outputs.

    image_output_dir_path: str
        A string that provides the DIRECTORY path (including "/" character at the end) to store image outputs.

    Returns
    ---------
    None
    
    Examples
    ---------
    main(
        train_data_file_path="./data/processed/train.csv",
        test_data_file_path="./data/processed/test.csv",
        csv_output_dir_path="./data/output/",
        image_output_dir_path="./img/"
    )

    """

    if not train_data_file_path:
        raise Exception("Please provide a valid file path for training data.")
    
    if not test_data_file_path:
        raise Exception("Please provide a valid file path for test data.")
    
    if csv_output_dir_path[-1] != "/":
        raise Exception("Please include the '/' character at the end of the csv_output_dir_path")

    if image_output_dir_path[-1] != "/":
        raise Exception("Please include the '/' character at the end of the image_output_dir_path")


    # Training Data
    train_data  = pd.read_csv(train_data_file_path)

    X_train = train_data.drop(["G3", "G2", "G1"], axis = 1)
    y_train = train_data["G3"]

    # Identify numerical vs categorical features
    categorical_features = X_train.loc[:,("school","sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", 
                                    "guardian","schoolsup", "famsup", "paid","activities","nursery", "higher", 
                                    "internet","romantic")].columns

    numeric_features = X_train.loc[:,("age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", 
                                        "freetime", "goout", "Dalc", "Walc", "health", "absences")].columns

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('ohe', OneHotEncoder(drop = "first"), categorical_features)
        ])

    # Convert to dataframe
    X_train_trans = pd.DataFrame(preprocessor.fit_transform(X_train),
                                index = X_train.index,
                                columns = (list(numeric_features) +
                                        list(preprocessor.named_transformers_['ohe'].get_feature_names(categorical_features))))

    ######## Modelling Begins ########
     # Linear Model Lasso with BayesOpt
    def cv_mse_lmlasso(alpha):
        """ 
        Performs cross validation for LM regressor with Lasso regression. To be used for Bayesian optimiser maximizer function.
        
        Parameters
        ----------
        alpha : float
            L1 regularisation constant

        Returns
        -------
        float
            Cross validation score based on negative mean squared error.
            
        """
        estimator = Lasso(alpha)

        # Note that scoring is neg_mean_squared_error, which means higher the score, the better the model
        return cross_validate(estimator, X_train_trans, y_train, cv = 10, scoring = "neg_root_mean_squared_error")["test_score"].mean()
    
    lmlasso_params = {'alpha':(0.001,100)}
    optimizer_lmlasso = BayesianOptimization(cv_mse_lmlasso, lmlasso_params, random_state = 1)
    print("HyperParameter Tuning: Linear Regression with Ridge")
    optimizer_lmlasso.maximize(n_iter = 10)

    # Linear Model Ridge with BayesOpt
    def cv_mse_lmridge(alpha):
        """ 
        Performs cross validation for LM regressor with Ridge regression. To be used for Bayesian optimiser maximizer function.
        
        Parameters
        ----------
        alpha : float
            L2 regularisation constant

        Returns
        -------
        float
            Cross validation score based on negative mean squared error.
            
        """
        estimator = Ridge(alpha)

        # Note that scoring is neg_mean_squared_error, which means higher the score, the better the model
        return cross_validate(estimator, X_train_trans, y_train, cv = 10, scoring = "neg_root_mean_squared_error")["test_score"].mean()

    lmridge_params = {'alpha':(0.001,100)}
    optimizer_lmridge = BayesianOptimization(cv_mse_lmridge, lmridge_params, random_state= 1)
    print("HyperParameter Tuning: Linear Regression with Ridge")
    optimizer_lmridge.maximize(n_iter = 10)

    # SKLearn Random Forest with BayesOpt
    def cv_mse_rf(n_estimators,max_depth, max_features):
        """ 
        Performs cross validation for Random Forest Regressor. To be used for Bayesian optimiser maximizer function.
        
        Parameters
        ----------
        n_estimators : float
            Number of estimators for random forest
        max_depth : float
            Max depth of trees in random forest
        max_features : float
            Max number of features in random forest

        Returns
        -------
        float
            Cross validation score based on negative mean squared error.
            
        """
        # Convert chosen hyperparams to discrete integer
        max_depth = int(max_depth)
        max_features = int(max_features)
        n_estimators = int(n_estimators)
        
        estimator = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)

        # Note that scoring is neg_mean_squared_error, which means higher the score, the better the model
        return cross_validate(estimator, X_train_trans, y_train, cv = 10, scoring = "neg_root_mean_squared_error")["test_score"].mean()

    rf_params = {'n_estimators':(10,150), 'max_depth':(10,200), 'max_features':(2, 30)}
    optimizer_rf = BayesianOptimization(cv_mse_rf, rf_params, random_state= 1)
    print("HyperParameter Tuning: Random Forest Regressor")
    optimizer_rf.maximize(n_iter = 20)

    # XGBoost Regressor with BayesOpt
    def cv_mse_xgb(n_estimators, max_depth, learning_rate, subsample, gamma, reg_alpha, reg_lambda):
        """ 
        Performs cross validation for Random Forest Regressor. To be used for Bayesian optimiser maximizer function.
        
        Parameters
        ----------
        n_estimators : float
            Number of estimators
        max_depth : float
            Max depth of trees
        learning_rate : float
            Learning rate
        subsample : float
            Subsample ratio of training instances 
        gamma : float
            Min loss reduction to make further partition on leaf node   
        reg_alpha : float
            L1 regularisation
        reg_lambda : float
            L2 regularisation

        Returns
        -------
        float
            Cross validation score based on negative mean squared error.
            
        """
        # Convert chosen hyperparams to discrete integer
        max_depth = int(max_depth)
        n_estimators = int(n_estimators)
        
        estimator = xgb.XGBRegressor(objective='reg:squarederror',
                                    n_estimators = n_estimators, 
                                    max_depth = max_depth, 
                                    learning_rate = learning_rate, 
                                    subsample = subsample,
                                    gamma = gamma, 
                                    reg_alpha = reg_alpha, 
                                    reg_lambda = reg_lambda)

        # Note that scoring is neg_mean_squared_error, which means higher the score, the better the model
        return cross_validate(estimator, X_train_trans, y_train, cv = 10, scoring = "neg_root_mean_squared_error")["test_score"].mean()

    xgb_params = {'n_estimators':(10, 150), 'max_depth':(10, 200), 'learning_rate':(0, 1),
                'subsample':(0, 1), 'gamma':(0, 50), 'reg_alpha':(0, 100), 'reg_lambda':(0, 100)}
    # Warnings due to some current issue with xgboost incompatibility with pandas deprecation
    # Fix will be for upcoming xgboost version 1.0.0, but latest version is only 0.90
    # See https://github.com/dmlc/xgboost/issues/4300
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    optimizer_xgb = BayesianOptimization(cv_mse_xgb, xgb_params, random_state = 1)
    print("HyperParameter Tuning: XGBoost Regressor")
    optimizer_xgb.maximize(n_iter = 20)

    # LightGBM with BayesOpt
    def cv_mse_lgbm(n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda):
        """ 
        Performs cross validation for Random Forest Regressor. To be used for Bayesian optimiser maximizer function.
        
        Parameters
        ----------
        n_estimators : float
            Number of estimators
        max_depth : float
            Max depth of trees
        learning_rate : float
            Learning rate
        reg_alpha : float
            L1 regularisation
        reg_lambda : float
            L2 regularisation

        Returns
        -------
        float
            Cross validation score based on negative mean squared error.
            
        """
        # Convert chosen hyperparams to discrete integer
        max_depth = int(max_depth)
        n_estimators = int(n_estimators)
        
        estimator = lgbm.LGBMRegressor(n_estimators = n_estimators, 
                                    max_depth = max_depth, 
                                    learning_rate = learning_rate, 
                                    reg_alpha = reg_alpha, 
                                    reg_lambda = reg_lambda)
        
        # Note that scoring is neg_mean_squared_error, which means higher the score, the better the model
        return cross_validate(estimator, X_train_trans, y_train, cv = 10, scoring = "neg_root_mean_squared_error")["test_score"].mean()

    lgbm_params = {'n_estimators':(10, 150), 'max_depth':(10, 200), 'learning_rate':(0.001, 1),
                'reg_alpha':(0, 100), 'reg_lambda':(0, 100)}
    optimizer_lgbm = BayesianOptimization(cv_mse_lgbm, lgbm_params, random_state = 1)
    print("HyperParameter Tuning: LGBM Regressor")
    optimizer_lgbm.maximize(n_iter = 20)

    # Compare CV Scores across the best models
    cv_rmse = [-optimizer_lmlasso.max['target'], 
                -optimizer_lmridge.max['target'], 
                -optimizer_rf.max['target'],
                -optimizer_xgb.max['target'],
                -optimizer_lgbm.max['target']]

    models = ["lm_lasso", "lm_ridge", "randomforest", "xgb", "lgbm"]

    cv_df = pd.DataFrame(cv_rmse, index = models, columns = ["cv_score"])

    # Output CV_df to csv
    cv_df.to_csv(csv_output_dir_path+"/cv_results.csv")

    # Adjusting discrete hyperparam for certain models
    lmlasso_hyperparam = optimizer_lmlasso.max['params']

    lmridge_hyperparam = optimizer_lmridge.max['params']

    rf_hyperparam = optimizer_rf.max['params']
    rf_hyperparam['max_depth'] = int(rf_hyperparam['max_depth'])
    rf_hyperparam['max_features'] = int(rf_hyperparam['max_features'])
    rf_hyperparam['n_estimators'] = int(rf_hyperparam['n_estimators'])

    xgb_hyperparam = optimizer_xgb.max['params']
    xgb_hyperparam['max_depth'] = int(xgb_hyperparam['max_depth'])
    xgb_hyperparam['n_estimators'] = int(xgb_hyperparam['n_estimators'])

    lgbm_hyperparam = optimizer_lgbm.max['params']
    lgbm_hyperparam['max_depth'] = int(lgbm_hyperparam['max_depth'])
    lgbm_hyperparam['n_estimators'] = int(lgbm_hyperparam['n_estimators'])

    # Store as Series for writing to csv.
    lmlasso_hyperparam_series = pd.Series(optimizer_lmlasso.max['params'])
    lmridge_hyperparam_series = pd.Series(optimizer_lmridge.max['params'])
    rf_hyperparam_series = pd.Series(rf_hyperparam)
    xgb_hyperparam_series = pd.Series(xgb_hyperparam)
    lgbm_hyperparam_series = pd.Series(lgbm_hyperparam)

    # Output model params to csv
    lmlasso_hyperparam_series.to_csv(csv_output_dir_path+"lmlasso_hyperparam.csv", header = False)
    lmridge_hyperparam_series.to_csv(csv_output_dir_path+"lmridge_hyperparam.csv", header = False)
    rf_hyperparam_series.to_csv(csv_output_dir_path+"rf_hyperparam.csv", header = False)
    xgb_hyperparam_series.to_csv(csv_output_dir_path+"xgb_hyperparam.csv", header = False)
    lgbm_hyperparam_series.to_csv(csv_output_dir_path+"lgbm_hyperparam.csv", header = False)

    # # Optional: Read in stored hyperparams. To be used when restart from offline
    # # Read in stored hyperparams.
    # lmlasso_hyperparam_series = pd.Series.from_csv("./data/output/lmlasso_hyperparam.csv")
    # lmridge_hyperparam_series = pd.Series.from_csv("./data/output/lmridge_hyperparam.csv")
    # rf_hyperparam_series = pd.Series.from_csv("./data/output/rf_hyperparam.csv")
    # xgb_hyperparam_series = pd.Series.from_csv("./data/output/xgb_hyperparam.csv")
    # lgbm_hyperparam_series = pd.Series.from_csv("./data/output/lgbm_hyperparam.csv")

    # # Reconfigure hyperparams due to float64 conversion for certain integer hyperparameters
    # lmlasso_hyperparam = dict()
    # for i in lmlasso_hyperparam_series.index:
    #     lmlasso_hyperparam[i] = lmlasso_hyperparam_series[i]

    # lmridge_hyperparam = dict()
    # for i in lmridge_hyperparam_series.index:
    #     lmridge_hyperparam[i] = lmridge_hyperparam_series[i]

    # rf_hyperparam = dict()
    # for i in rf_hyperparam_series.index:
    #     rf_hyperparam[i] = rf_hyperparam_series[i]

    # xgb_hyperparam = dict()
    # for i in xgb_hyperparam_series.index:
        
    #     # For integer hyperparams
    #     if i == "max_depth" or i == "n_estimators" or i == "max_features":
    #         xgb_hyperparam[i] = int(xgb_hyperparam_series[i])
    #     # For float hyperparams
    #     else:
    #         xgb_hyperparam[i] = xgb_hyperparam_series[i]

    # lgbm_hyperparam = dict()
    # for i in lgbm_hyperparam_series.index:
        
    #     # For integer hyperparams
    #     if i == "max_depth" or i == "n_estimators" or i == "max_features":
    #         lgbm_hyperparam[i] = int(lgbm_hyperparam_series[i])
    #     # For float hyperparams
    #     else:
    #         lgbm_hyperparam[i] = lgbm_hyperparam_series[i]

    # Create Models for Test Scoring
    best_lasso = Lasso(random_state = 1).set_params(**lmlasso_hyperparam)
    best_lasso.fit(X_train_trans, y_train)

    best_ridge = Ridge(random_state = 1).set_params(**lmridge_hyperparam)
    best_ridge.fit(X_train_trans, y_train)

    best_rf = RandomForestRegressor(random_state = 1).set_params(**rf_hyperparam)
    best_rf.fit(X_train_trans, y_train)

    best_xgb = xgb.XGBRegressor(random_state = 1).set_params(**xgb_hyperparam)
    best_xgb.fit(X_train_trans, y_train)

    best_lgbm = lgbm.LGBMRegressor(random_state = 1).set_params(**lgbm_hyperparam)
    best_lgbm.fit(X_train_trans, y_train)

    # Test set
    test_data  = pd.read_csv(test_data_file_path)

    X_test = test_data.drop(["G3", "G2", "G1"], axis = 1)
    y_test = test_data["G3"]

    # Convert to dataframe with preprocessor
    X_test_trans = pd.DataFrame(preprocessor.fit_transform(X_test),
                                index = X_test.index,
                                columns = (list(numeric_features) +
                                        list(preprocessor.named_transformers_['ohe'].get_feature_names(categorical_features))))

    # Test Scoring
    test_rmse = []
    test_rmse.append(np.sqrt(mean_squared_error(y_test, best_lasso.predict(X_test_trans))))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, best_ridge.predict(X_test_trans))))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, best_rf.predict(X_test_trans))))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, best_xgb.predict(X_test_trans))))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, best_lgbm.predict(X_test_trans))))

    # Convert results to Dataframe
    test_rmse = pd.DataFrame(test_rmse, index= models, columns = ["test_rmse"])
    test_rmse = test_rmse.sort_values(by="test_rmse", ascending = True)

    # Output to csv
    test_rmse.to_csv(csv_output_dir_path+"final_results.csv")

    # Create dictionary of all models
    models_dict = dict()
    models_dict["lm_lasso"] = best_lasso
    models_dict["lm_ridge"] = best_ridge
    models_dict["randomforest"] = best_rf
    models_dict["xgb"] = best_xgb
    models_dict["lgbm"] = best_lgbm

    # Choose best model
    best_model = models_dict[list(test_rmse.head(1).index)[0]]

    # If model is linear regression model, use "coef_" to extract weights
    if (best_model == best_lasso) or (best_model == best_ridge):
        feat_importance = pd.DataFrame(best_model.coef_, index = X_train_trans.columns, columns = ["Importance"])

    # If model is tree based, use "feature_importances" to extract importances    
    else: 
        feat_importance = pd.DataFrame(best_model.feature_importances_, index = X_train_trans.columns, columns = ["Importance"])

    # Sort feat_importance by descending order
    feat_importance = feat_importance.sort_values(by = "Importance", ascending = False).reset_index()

    # Output feat_importance model
    feat_importance.to_csv(csv_output_dir_path+"feat_importance.csv")

    # Altair Plot of Lollipop Chart
    # https://github.com/nipunbatra/50-ggplot-python/blob/master/Altair/DivergingLollipop.ipynb

    # Lollipop bar/sticks
    c1 = alt.Chart(feat_importance.head(10)).mark_bar(color='pink', size = 5).encode(
        y=alt.Y('index', sort=alt.EncodingSortField(order='descending', field='Importance'), title = None),
        x=alt.X('Importance')
    )

    # Lollipop Heads/Circles
    c2 = alt.Chart(feat_importance.head(10)).mark_circle(color='lightblue', size=1200).encode(
        y=alt.Y('index', sort=alt.EncodingSortField(order='descending', field='Importance')),
        x=alt.X('Importance' ), 
        text='Importance'
    )

    # Lollipop Text/Importance Weights
    c3 = alt.Chart(feat_importance.head(10)).mark_text(color='black').encode(
        y=alt.Y('index', sort=alt.EncodingSortField(order='descending', field='Importance')),
        x=alt.X('Importance' ), 
        text='Importance'
    )

    # Create layered chart
    chart = alt.layer(c1,c2,c3)

    # Configure chart size and output png file
    chart.configure(
        numberFormat="0.4f"
    ).properties(
        title = "Top 10 Features Ranked According to Importance ("+list(test_rmse.head(1).index)[0]+")",
        width = 800,
        height = 400
    ).save(image_output_dir_path+"ranked_features.png")

def check_train_data_file_path():
    """
    This function checks if an exception is raised if an empty train_data_file_path is provided to 
    main.
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    None, if the test has passed, and a Failed message if the test has not passed.
    
    Examples
    ----------
    check_train_data_file_path()
    
    """
    with pytest.raises(Exception):
        main(
            test_data_file_path="./data/processed/test.csv",
            csv_output_dir_path="./data/output/",
            image_output_dir_path="./img/output/"
        )
        
assert check_train_data_file_path() == None, "Invalid train_data_file_path of main function has failed."

if __name__ == "__main__":
    main(opt["--train_data_file_path"], opt["--test_data_file_path"], opt["--csv_output_dir_path"], opt["--image_output_dir_path"])
