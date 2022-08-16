import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evauation_model(pred, y_val):
  score_MSE = round(mean_squared_error(pred, y_val),2)
  score_MAE = round(mean_absolute_error(pred, y_val),2)
  score_r2score = round(r2_score(pred, y_val),2)
  return score_MSE, score_MAE, score_r2score


pound_to_kg = 0.45
inches_to_meters = 0.0254

data = pd.read_csv("/Users/pranay/Desktop/Prudential/Dummy-Data.csv")

data['BMI'] = data['Wt']*pound_to_kg/data['Ht'].map(lambda x : ((int(str(x)[0])*12 + int(str(x)[-2:]))*inches_to_meters)**2)


def category(age,bmi,gender):
    if gender=='Male':
        if (age>=18 and age<=39) and (bmi<17.49 or bmi>38.5):
            return 750
        elif (age>=40 and age<=59) and (bmi<18.49 or bmi>38.5):
            return 1000
        elif (age>=60) and (bmi<18.49 or bmi>38.5):
            return 2000
        else:
            return 500
    else:
        return category(age,bmi,'Male')*(1-0.1)
    

data_cleaned = data.drop(['IssueDate','AppID'], axis=1)


data['quote'] = data.apply(lambda x: category(x['Ins_Age'],x['BMI'],x['Ins_Gender']),axis=1)

reason_code_dictionary = {750:"Age is between 18 to 39 and 'BMI' is either less than 17.49 or greater than 38.5",
               1000:"Age is between 40 to 59 and 'BMI' is either less than 18.49 or greater then 38.5”",
               2000:"Age is greater than 60 and 'BMI' is either less than 18.49 or greater than 38.5",
               500:"BMI is in right range"}



def application(Ins_Age, Ins_Gender, Ht, Wt):
    
    test_df = pd.DataFrame([[Ins_Age, Ins_Gender, Ht, Wt]],columns = ['Ins_Age', 'Ins_Gender', 'Ht', 'Wt'])
    
    test_df['BMI'] = test_df['Wt']*pound_to_kg/test_df['Ht'].map(lambda x : ((int(str(x)[0])*12 + int(str(x)[-2:]))*inches_to_meters)**2)

    x = test_df.iloc[0]
    
    quote = category(x['Ins_Age'],x['BMI'],x['Ins_Gender'])
    
    for_reason_code_quote = category(x['Ins_Age'],x['BMI'],'Male')

    
    return quote, reason_code_dictionary[for_reason_code_quote]


def models_score(model_name, train_data, y_train, val_data, y_val):
    model_list = ["Decision_Tree", "Random_Forest", "XGboost_Regressor"]
    # model_1
    if model_name == "Decision_Tree":
        reg = DecisionTreeRegressor(random_state=42)
    # model_2
    elif model_name == "Random_Forest":
        reg = RandomForestRegressor(random_state=42)

    # model_3
    elif model_name == "XGboost_Regressor":
        reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, )
    else:
        print("please enter correct regressor name")

    if model_name in model_list:
        reg.fit(train_data, y_train)
        pred = reg.predict(val_data)

        score_MSE, score_MAE, score_r2score = evauation_model(pred, y_val)
        return round(score_MSE, 2), round(score_MAE, 2), round(score_r2score, 2)


X = data[['Ins_Age', 'Ht', 'Wt','quote', 'Ins_Gender']]
y = data['BMI']   
#%%
X_train, X_test, y_train, y_test = train_test_split(data_cleaned,y, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
X_train['Ins_Gender'] = label_encoder.fit_transform(X_train['Ins_Gender'].values)
X_test['Ins_Gender'] = label_encoder.transform(X_test['Ins_Gender'].values)
model_list = ["Decision_Tree","Random_Forest","XGboost_Regressor"]
#%%
result_scores = []
for model in model_list:
    score = models_score(model, X_train, y_train, X_test, y_test)
    result_scores.append((model, score[0], score[1],score[2]))
    print(model,score)

df_result_scores = pd.DataFrame(result_scores,columns=["model","mse","mae","r2score"])
df_result_scores
#%%
num_estimator = [100, 150, 200, 250]

space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
         'gamma': hp.uniform('gamma', 1, 9),
         'reg_alpha': hp.quniform('reg_alpha', 30, 180, 1),
         'reg_lambda': hp.uniform('reg_lambda', 0, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
         'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
         'n_estimators': hp.choice("n_estimators", num_estimator),
         }


def hyperparameter_tuning(space):
    model = xgb.XGBRegressor(n_estimators=space['n_estimators'], max_depth=int(space['max_depth']),
                             gamma=space['gamma'],
                             reg_alpha=int(space['reg_alpha']), min_child_weight=space['min_child_weight'],
                             colsample_bytree=space['colsample_bytree'], objective="reg:squarederror")

    score_cv = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()
    return {'loss': -score_cv, 'status': STATUS_OK, 'model': model}


trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print(best)
#%%
best['max_depth'] = int(best['max_depth']) # convert to int
best["n_estimators"] = num_estimator[best["n_estimators"]] # assing n_estimator because it returs the index
best_xgboost_model = xgb.XGBRegressor(**best)
best_xgboost_model.fit(X_train,y_train)
pred = best_xgboost_model.predict(X_test)
score_MSE, score_MAE, score_r2score = evauation_model(pred,y_test)
to_append = ["XGboost_hyper_tuned",score_MSE, score_MAE, score_r2score]
df_result_scores.loc[len(df_result_scores)] = to_append

best_xgboost_model.save_model("best_model.json")