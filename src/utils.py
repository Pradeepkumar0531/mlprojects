import sys
import os
import dill
import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, Y_train, X_test, Y_test, models,param):
    try:
        report = {}
        for model_name, model in models.items():
            
            para=param[model_name]

            rs=RandomizedSearchCV(model,para,cv=3)
            rs.fit(X_train,Y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train, Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

            train_model_score = r2_score(Y_train, Y_train_pred)
            test_model_score = r2_score(Y_test, Y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)