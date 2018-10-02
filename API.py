# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 12:53:35 2018

@author: ishan-adarsh
"""

import flask
from flask import request, jsonify, render_template
from werkzeug import secure_filename
import numpy as np
import pandas as pd
import pickle
import json
import os
import datetime
from flask_cors import CORS

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True
file_location = "C:\\Users\\ishan-adarsh\\Desktop\\ML\\Predict IT Code\\"
upload_location = file_location + 'TrainingData\\'


# =============================================================================
# Fuction to Retrain the Model
# =============================================================================
@app.route('/api/v1/resources/uploader', methods = ['GET', 'POST'])
def upload_file():

    format = "%Y%m%dT%H%M%S"
    now = datetime.datetime.utcnow().strftime(format)
    
    if request.method == 'POST':
      f = request.files['file']
      filename = now + '_' + f.filename
      f.save(os.path.join(upload_location, filename))
      
      #Step 1 Loading Data
      dataset = pd.read_csv(upload_location + '\\' +filename)
      X = dataset.iloc[:,:-1].values
      Y =dataset.iloc[:,[-1]].values
      Y_Encoded=dataset.iloc[:,[-1]].values
      
      #Step 2 Implementing Label Encoding
      from sklearn.preprocessing import LabelEncoder
      X_labelEncoder= LabelEncoder()
      X[:,0]=X_labelEncoder.fit_transform(X[:,0])
      
      from sklearn.preprocessing import LabelEncoder
      Y_labelEncoder= LabelEncoder()
      Y = Y_labelEncoder.fit_transform(Y[:,0])
        
      dict_Mercer_Job_Library_Code = {} 
      for i in range(len(Y)):
          dict_Mercer_Job_Library_Code[Y[i]] = Y_Encoded[i,0]
      
      #Step 3 Spliting the dataset into 80 % train set and 20% test set
      from sklearn.cross_validation import train_test_split
      X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=6)
      
      #Step 4 Processing Data
      from sklearn.preprocessing import StandardScaler
      independent_scaler = StandardScaler()
      X_train = independent_scaler.fit_transform(X_train)
      X_test = independent_scaler.transform(X_test)
      
      #Step 5 Transforming the data
      from sklearn.decomposition import PCA
      pca = PCA(n_components = 4)
      pca.fit(X_train)   
     
      principalComponents = pca.transform(X_train)
      principalComponentstest = pca.transform(X_test)
      
      #Step 6 Implementing Random Forest
      from sklearn.ensemble import RandomForestClassifier
      RFModel = RandomForestClassifier(warm_start=True, n_estimators = 11, random_state = 3)
      RFFit = RFModel.fit(principalComponents, Y_train)
      
      #Step 7 Predicting from the created model
      predictionValue = RFModel.predict(principalComponentstest)
      
      #Step 8 Calculating the Model Accuracy
      from sklearn import metrics
      accuracy_score = metrics.accuracy_score(Y_test,predictionValue)    
    
      
      #Step 9 Saving the objects for Prediction Analysis
      pkl_filename = file_location + "\\Models\\" + request.form['filename'] +".pkl"
      with open(pkl_filename, 'wb') as file:  
          pickle.dump(RFModel, file)
            
      independent_scaler_filename = file_location + "independent_scaler.pkl"
      with open(independent_scaler_filename, 'wb') as file:
          pickle.dump(independent_scaler, file)
        
      pca_filename = file_location + "pca.pkl"
      with open(pca_filename, 'wb') as file:
          pickle.dump(pca, file)
            
      RFclassifier_filename = file_location + "RFclassifier.pkl"
      with open(RFclassifier_filename, 'wb') as file:
          pickle.dump(RFFit, file)
    
      dict_Mercer_Job_Library_Code_filename = file_location + "dict_Mercer_Job_Library_Code.pkl"
      with open(dict_Mercer_Job_Library_Code_filename, 'wb') as file:
          pickle.dump(dict_Mercer_Job_Library_Code, file)

      jsonResult = json.dumps({"accuracy" : accuracy_score})
      return jsonify(jsonResult)

# =============================================================================
# Function to Predict the output as per the Data entered by the user		
# =============================================================================
@app.route('/api/v1/resources/predict', methods=['POST'])
def api_predict():   
   
    arr = []
    arr.append([request.json['Location_City'],
                request.json['Median_Annual_Salary'],
                request.json['Career_Level'],
                request.json['Organization_Name'],              
                request.json['Organization_Job_Code'],
                request.json['Expertise_Score'],
                request.json['Judgement_Score'],
                request.json['Accountability_Score'],               
                request.json['Organization_Supervisor_Title_Code'],
                request.json['Average_Work_Experience']                
                ])
    
    #Step 1 Loading Data
    X_Predict = pd.DataFrame(arr)    
    
    #Step 2 Processing Data
    X_Predict = process_data(X_Predict)   
    
    #Step 3 Implementing Model
    result = implement_model(X_Predict)
    result = result.tolist()[0]
    #Pulling the value fom Dictionary    
    pkl_filename = file_location + 'dict_Mercer_Job_Library_Code.pkl'   
    with open(pkl_filename, 'rb') as file:  
        dict_Mercer_Job_Library_Code = pickle.load(file)   
    result = dict_Mercer_Job_Library_Code.get(result)
    
    #Step 4 Calculate Model Prediction Percentage
    predictionsPercentage = model_prediction_percentage(X_Predict)
   
    #Step 5 Preparing the output   	
    result = {'Mercer_Job_Library_Code' : result, 'Predictions_Percentage' : predictionsPercentage}  
    jsonResult = json.dumps(result)
    return jsonify(jsonResult)

#Data PreProcessing
def process_data(X_Predict): 
     #==============================================================
    #Label Encoding for the Location City Column
    from sklearn.preprocessing import LabelEncoder
    X_labelEncoder= LabelEncoder()
    X_Predict.iloc[:,0]=X_labelEncoder.fit_transform(X_Predict.iloc[:,0])
    
     # Load from file for Scaling the data
    pkl_filename = file_location + 'independent_scaler.pkl'   
    with open(pkl_filename, 'rb') as file:  
        independent_scaler = pickle.load(file)    
    X_Predict = independent_scaler.transform(X_Predict)


    # Load from file for performing Principle Component Analysis
    pkl_filename = file_location + 'pca.pkl'
    with open(pkl_filename, 'rb') as file:  
        pca = pickle.load(file)   
    X_Predict = pca.transform(X_Predict)    
    return X_Predict

#Model Implementation
def implement_model(X_Predict):
    #==============================================================
    #Loading Model
    pkl_filename = file_location + '\\Models\\' +  request.json['Model']
    # Load from file
    with open(pkl_filename, 'rb') as file:  
        pickle_model = pickle.load(file)
    
    result = pickle_model.predict(X_Predict)
    return result

#Calculate Model Prediction Percentage
def model_prediction_percentage(X_Predict):
    #==============================================================
    #Loading Model's Prediction Percentage Object
    pkl_filename = file_location + 'RFclassifier.pkl'
    # Load from file
    with open(pkl_filename, 'rb') as file:  
        RFclassifier = pickle.load(file)
    
    predictionsPercentage = RFclassifier.predict_proba(X_Predict)
     #Pulling the value fom Dictionary    
    pkl_filename = file_location + 'dict_Mercer_Job_Library_Code.pkl'   
    with open(pkl_filename, 'rb') as file:  
        dict_Mercer_Job_Library_Code = pickle.load(file)   
    #Initializing value for pulling Probility Index Value    
    i = 0
    pridict_proba = {}
    for val in predictionsPercentage[0]:
        name = dict_Mercer_Job_Library_Code.get(i)
        pridict_proba[name] = val
        i+=1
        
    predictionsPercentage = pridict_proba
    return predictionsPercentage


@app.route('/api/v1/resources/getModels', methods = ['POST'])
def getModels():
    filelist = os.listdir(file_location + "\\Models\\")
    dict_filelist = {}
    for file in filelist:
         dict_filelist[os.path.splitext(file)[0]] = file
    jsonResult = json.dumps(dict_filelist)
    return jsonify(jsonResult)
#request.json['ishan']
app.run()
