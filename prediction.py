import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
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

data_cleaned = data.drop(['IssueDate','AppID'], axis=1)
X = data[['Ins_Age', 'Ht', 'Wt','quote', 'Ins_Gender']]
y = data['BMI']  
X_train, X_test, y_train, y_test = train_test_split(data_cleaned,y, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
X_train['Ins_Gender'] = label_encoder.fit_transform(X_train['Ins_Gender'].values)
X_test['Ins_Gender'] = label_encoder.transform(X_test['Ins_Gender'].values)
#save label encoder classes
np.save('classes.npy', label_encoder.classes_)



# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")
pred = best_xgboost_model.predict(X_test)
score_MSE, score_MAE, score_r2score = evauation_model(pred, y_test)
print(score_MSE, score_MAE, score_r2score)
#%%
loaded_encoder = LabelEncoder()
loaded_encoder.classes_ = np.load('classes.npy',allow_pickle=True)
print(X_test.shape)

########
input_person = loaded_encoder.transform(np.expand_dims("Male",-1))
print(int(input_person))
inputs = np.expand_dims([int(input_person),35,'Make',510,200],0)
print(inputs.shape)
prediction = best_xgboost_model.predict(inputs)
print("final pred", np.squeeze(prediction,-1))