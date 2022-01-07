#!/usr/bin/env python
# coding: utf-8

# # Importing necessary modules

# In[74]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.preprocessing import LabelEncoder


# # Preprocessing data

# In[75]:


def telcoDataPreprocess():
    df = pd.read_csv (r'Telco-Customer-Churn.csv')

    df.gender[df.gender == 'Male'] = 1
    df.gender[df.gender == 'Female'] = 0

    categories = ['Partner','Dependents','PhoneService', 'PaperlessBilling']
    for cat in categories:
        df[cat][df[cat] == 'Yes'] = 1
        df[cat][df[cat] == 'No'] = 0
        
    #making output within -1 to 1 as tanh is the logistic regression function
    df['Churn'][df['Churn'] == 'Yes'] = 1
    df['Churn'][df['Churn'] == 'No'] = -1
    
#     print(df['Churn'].unique())

    #converting data type from string to float
    dummy = []
    for i in range(len(df['TotalCharges'])):
        if df['TotalCharges'][i] != " ":
            df['TotalCharges'][i] = float(df['TotalCharges'][i])
            dummy.append(df['TotalCharges'][i])
            
#     print(dummy)
    avg = sum(dummy) / len(dummy)
    print(avg)

    for i in range(len(df['TotalCharges'])):
        if df['TotalCharges'][i] == " ":
            df['TotalCharges'][i] = avg
        
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])



    #min-max scaling or normalization
    normalize_cats = ['tenure', 'MonthlyCharges', 'TotalCharges']

    for col in normalize_cats:
        norm = df[[col]]
        norm = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(norm))
        df[col] = norm[0]

    #removing customer ID as it's a unique string
    df.drop(['customerID'], axis = 1, inplace = True)


    #doing one hot encoding where unique values are more than two in a particular column
    onehot_cols = []
    for col in df.columns:
        if len(list(df[col].unique())) >= 3:
            if isinstance(df[col][0], str):
                onehot_cols.append(col)

    
    for i in range(0, len(onehot_cols)):
        dummies = pd.get_dummies(df[[onehot_cols[i]]])
        res = pd.concat([df, dummies], axis = 1)
        df = res.drop(columns=[onehot_cols[i]])


    # print(df)
    ##somehow last column is not churn at this moment, so making it the last column
    dfp = df.pop('Churn') # remove column output and store it in dfp
    df['Churn']=dfp # add churn as a 'new' column at te end.

#     print (df.iloc[:,-1])


    #preparing training and testing datasets
    df1 = df.copy()
    df2 = df.copy()
    df_features = df1.iloc[:,:-1]
    df_output = df2.iloc[:,-1]
    
    features_training_data = df_features.sample(frac=0.8, random_state=25)
    features_testing_data = df_features.drop(features_training_data.index)

    output_training_data = df_output.sample(frac=0.8, random_state=25)
    output_testing_data = df_output.drop(output_training_data.index)


    features_training_data = features_training_data.to_numpy()
    features_testing_data = features_testing_data.to_numpy()

    output_training_data = output_training_data.to_numpy()
    output_testing_data = output_testing_data.to_numpy()


    features_training_data = features_training_data.astype('float')
    features_testing_data = features_testing_data.astype('float')

    output_training_data = output_training_data.astype('int')
    output_testing_data = output_testing_data.astype('int')
    print(np.unique(np.array(output_testing_data)))
    

    return features_training_data, features_testing_data, output_training_data, output_testing_data


# In[76]:


features_train, features_test, output_train, output_test = telcoDataPreprocess()

print(features_train, features_test, output_train, output_test)


# In[77]:


def creditDataPreprocess():
    dfcred = pd.read_csv (r'creditcard.csv')


    #min-max scaling or normalization
    normalize_cats2 = ['Time', 'V1', 'V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']

    for col in normalize_cats2:
        norm2 = dfcred[[col]]
        norm2 = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(norm2))
        dfcred[col] = norm2[0]

    dfcred['Class'][dfcred['Class'] == 1] = 1
    dfcred['Class'][dfcred['Class'] == 0] = -1

    #preparing training and testing datasets
    dfcred1 = dfcred.copy()
    dfcred2 = dfcred.copy()
    dfcred_features = dfcred1.iloc[:,:-1]
    dfcred_output = dfcred2.iloc[:,-1]

    credfeatures_training_data = dfcred_features.sample(frac=0.8, random_state=25)
    credfeatures_testing_data = dfcred_features.drop(credfeatures_training_data.index)

    credoutput_training_data = dfcred_output.sample(frac=0.8, random_state=25)
    credoutput_testing_data = dfcred_output.drop(credoutput_training_data.index)


    credfeatures_training_data = credfeatures_training_data.to_numpy()
    credfeatures_testing_data = credfeatures_testing_data.to_numpy()

    credoutput_training_data = credoutput_training_data.to_numpy()
    credoutput_testing_data = credoutput_testing_data.to_numpy()


    credfeatures_training_data = credfeatures_training_data.astype('float')
    credfeatures_testing_data = credfeatures_testing_data.astype('float')

    credoutput_training_data = credoutput_training_data.astype('int')
    credoutput_testing_data = credoutput_testing_data.astype('int')

    print(np.unique(np.array(credoutput_testing_data)))

    print(dfcred2)

    return credfeatures_training_data, credfeatures_testing_data, credoutput_training_data, credoutput_testing_data


# In[78]:


credfeatures_train, credfeatures_test, credoutput_train, credoutput_test = creditDataPreprocess()

print(credfeatures_train, credfeatures_test, credoutput_train, credoutput_test)


# In[87]:


def adultDataPreprocess():
    dfadult_train = pd.read_csv (r'adult-train.csv')
    dfadult_test = pd.read_csv (r'adult-test.csv')
    
    dfadult_train.sex[dfadult_train.sex == ' Male'] = 1
    dfadult_train.sex[dfadult_train.sex == ' Female'] = 0
    
    dfadult_test.sex[dfadult_test.sex == ' Male'] = 1
    dfadult_test.sex[dfadult_test.sex == ' Female'] = 0

    #min-max scaling or normalization
    normalize_cats3 = ['age', 'fnlwgt', 'education-num','capital-gain','capital-loss','hours-per-week']

    for col in normalize_cats3:
        norm3 = dfadult_train[[col]]
        norm4 = dfadult_test[[col]]
        norm3 = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(norm3))
        norm4 = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(norm4))
        dfadult_train[col] = norm3[0]
        dfadult_test[col] = norm4[0]

    dfadult_train['salary-scale'][dfadult_train['salary-scale'] == " >50K"] = 1
    dfadult_train['salary-scale'][dfadult_train['salary-scale'] == " <=50K"] = -1
    
    dfadult_test['salary-scale'][dfadult_test['salary-scale'] == " >50K."] = 1
    dfadult_test['salary-scale'][dfadult_test['salary-scale'] == " <=50K."] = -1

    # print(dfadult_test['workclass'].mode()[0])
    ## replacing ? mark with the most seen value
    work = dfadult_train['workclass'].mode()[0]
    work2 = dfadult_test['workclass'].mode()[0]
    for i in range (len(dfadult_train['workclass'])):
        if dfadult_train['workclass'][i] == " ?":

            dfadult_train['workclass'][i] = work
     
    for i in range (len(dfadult_test['workclass'])):
        if dfadult_test['workclass'][i] == " ?":

            dfadult_test['workclass'][i] = work2

    occ = dfadult_train['occupation'].mode()[0]
    occ2 = dfadult_test['occupation'].mode()[0]
    for i in range (len(dfadult_train['occupation'])):
        if dfadult_train['occupation'][i] == " ?":

            dfadult_train['occupation'][i] = occ
            
    for i in range (len(dfadult_test['occupation'])):
        if dfadult_test['occupation'][i] == " ?":

            dfadult_test['occupation'][i] = occ2


    country = dfadult_train['native-country'].mode()[0]
    country2 = dfadult_test['native-country'].mode()[0]
    for i in range (len(dfadult_train['native-country'])):
        if dfadult_train['native-country'][i] == " ?":

            dfadult_train['native-country'][i] = country
            
    for i in range (len(dfadult_test['native-country'])):
        if dfadult_test['native-country'][i] == " ?":

            dfadult_test['native-country'][i] = country2



    #doing label encoding (cause train and test are splitted) where unique values are more than two in a particular column
    label_cols3 = []
    for col in dfadult_train.columns:
        if len(list(dfadult_train[col].unique())) >= 3:
            if isinstance(dfadult_train[col][0], str):
                label_cols3.append(col)


    for i in label_cols3:
    

        c = dfadult_train[i]
        
        dfadult_train[i] = LabelEncoder().fit_transform(c)
        
        
#         dfadult_train[[i]] = LabelEncoder.fit_transform(dfadult_train[[i]])
    
    label_cols4 = []
    for col in dfadult_test.columns:
        if len(list(dfadult_test[col].unique())) >= 3:
            if isinstance(dfadult_test[col][0], str):
                label_cols4.append(col)


    for i in label_cols4:
        d = dfadult_test[i]
        dfadult_test[i] = LabelEncoder().fit_transform(d)

    # print(dfadult_test['native-country'].unique())

    ##somehow last column is not salary-scale at this moment, so making it the last column
    dfp3 = dfadult_train.pop('salary-scale') # remove column output and store it in dfp
    dfadult_train['salary-scale'] = dfp3 # add churn as a 'new' column at te end.
    
    ##somehow last column is not salary-scale at this moment, so making it the last column
    dfp4 = dfadult_test.pop('salary-scale') # remove column output and store it in dfp
    dfadult_test['salary-scale'] = dfp4 # add churn as a 'new' column at te end.

    print("column numbrs",dfadult_test.columns,dfadult_test.columns )


    #preparing training and testing datasets
    dfadtrain = dfadult_train.copy()
    dfadtest = dfadult_test.copy()
    dfadulttrain_features = dfadtrain.iloc[:,:-1]
    dfadulttrain_output = dfadtrain.iloc[:,-1]
    dfadulttest_features = dfadtest.iloc[:,:-1]
    dfadulttest_output = dfadtest.iloc[:,-1]
    
    print("printing shapes", dfadulttrain_features.shape, dfadulttest_features.shape)
    print(dfadulttrain_features)


    adultfeatures_training_data = dfadulttrain_features.to_numpy()
    adultfeatures_testing_data = dfadulttest_features.to_numpy()

    adultoutput_training_data = dfadulttrain_output.to_numpy()
    adultoutput_testing_data = dfadulttest_output.to_numpy()


    adultfeatures_training_data = adultfeatures_training_data.astype('float')
    adultfeatures_testing_data = adultfeatures_testing_data.astype('float')

    adultoutput_training_data = adultoutput_training_data.astype('int')
    adultoutput_testing_data = adultoutput_testing_data.astype('int')

    print(np.unique(np.array(adultoutput_testing_data)))


    return adultfeatures_training_data, adultfeatures_testing_data, adultoutput_training_data, adultoutput_testing_data


# In[88]:


adultfeatures_train, adultfeatures_test, adultoutput_train, adultoutput_test = adultDataPreprocess()

print(adultfeatures_train, adultfeatures_test, adultoutput_train, adultoutput_test)


# # Logistic regression parameters and other variables determining

# In[89]:


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    


# In[90]:


def derivative_tanh(x):
    return 1- (tanh(x)**2)


# In[91]:


def loss(y, y_hat):
    loss = (np.sum((y - y_hat) **2)) / len(y)
    return loss


# In[92]:


def gradients(X, y, y_hat, der_y):
    
    # X --> Input.
    # y --> true/target value.
    # y_hat --> hypothesis/predictions.
    # m-> number of training examples.
    
    m, n = X.shape
    
    # Gradient of loss w.r.t weights.
    for i in range(0, len(y_hat)):
        if y_hat[i] > 0:
            y_hat[i] = 1
            
        else:
            y_hat[i] = -1
            
    dw = (1/m)*np.dot(X.T, (y_hat - y)*der_y)
    
    # Gradient of loss w.r.t bias.
    db = (1/m)*np.sum((y_hat - y)) 
    
    return dw, db


# In[93]:


def predict(X, w, b):
    
    preds = tanh(np.dot(X, w) + b)
    
    # Empty List to store predictions.
    pred_class = []
    # if y_hat >= 0 --> round up to 1
    # if y_hat < 0 --> round up to 1
    pred_class = [1 if i > 0 else -1 for i in preds]
    
#     print(np.unique(np.array(pred_class)))
    
    return np.array(pred_class)


# In[94]:


def logistic_regression(features, output, bias, iterations, learning_rate):
    m, n = features.shape
    w = np.zeros((n,1))
    b = 0
    
    output = output.reshape(m,1)
    
    losses = []
    
    for i in range(iterations):

            
            # Calculating hypothesis/prediction.
        y_hat = tanh(np.dot(features, w) + b)
            
        der_y = derivative_tanh(np.dot(features, w) + b)
            
            # Getting the gradients of loss w.r.t parameters.
            
            
        dw, db = gradients(features, output, y_hat, der_y)
            
            
        w -= learning_rate*dw
        b -= learning_rate*db
        
        
        l = loss(output, tanh(np.dot(features, w) + b))
        losses.append(l)
    
    return w, b, losses
    


# In[95]:


# print(output_test)


# In[96]:


# Training 
w, b, l = logistic_regression(features_train, output_train, 0, 1000, 0.025)

# Testing
predicted_output = predict(features_test, w, b)

print(predicted_output)


# In[129]:


predicted_output2 = predict(features_train, w, b)


# In[97]:


wadult, badult, ladult = logistic_regression(adultfeatures_train, adultoutput_train, 0, 1000 , 0.025)
print(wadult.shape)
predicted_outputadult = predict(adultfeatures_test, wadult, badult)
print(predicted_outputadult)


# In[130]:


predicted_outputadult2 = predict(adultfeatures_train, wadult, badult)


# In[145]:


wcred, bcred, lcred = logistic_regression(credfeatures_train, credoutput_train, 0, 400 , 0.025)
predicted_outputcred = predict(credfeatures_test, wcred, bcred)
print(predicted_outputcred)


# In[146]:


predicted_outputcred2 = predict(credfeatures_train, wcred, bcred)


# In[127]:


print("Accuracy value of telco data on test data:", metrics.accuracy_score(output_test, predicted_output))
print("Precision value of telco data on test data:", metrics.precision_score(output_test, predicted_output))
print("Recall value of telco data on test data:", metrics.recall_score(output_test, predicted_output))
print("F1 score of telco data on test data:", metrics.f1_score(output_test, predicted_output))


# In[131]:


print("Accuracy value of telco data on train data:", metrics.accuracy_score(output_train, predicted_output2))
print("Precision value of telco data on train data:", metrics.precision_score(output_train, predicted_output2))
print("Recall value of telco data on train data:", metrics.recall_score(output_train, predicted_output2))
print("F1 score of telco data on train data:", metrics.f1_score(output_train, predicted_output2))


# In[101]:


print("Accuracy value of adult data on test data:", metrics.accuracy_score(adultoutput_test, predicted_outputadult))
print("Precision value of adult data on test data:", metrics.precision_score(adultoutput_test, predicted_outputadult))
print("Recall value of adult data on test data:", metrics.recall_score(adultoutput_test, predicted_outputadult))
print("F1 score of adult data on test data:", metrics.f1_score(adultoutput_test, predicted_outputadult))


# In[132]:


print("Accuracy value of adult data on train data:", metrics.accuracy_score(adultoutput_train, predicted_outputadult2))
print("Precision value of adult data on train data:", metrics.precision_score(adultoutput_train, predicted_outputadult2))
print("Recall value of adult data on train data:", metrics.recall_score(adultoutput_train, predicted_outputadult2))
print("F1 score of adult data on train data:", metrics.f1_score(adultoutput_train, predicted_outputadult2))


# In[147]:


print("Accuracy value of credit data on test data:", metrics.accuracy_score(credoutput_test, predicted_outputcred))
print("Precision value of credit data on test data:", metrics.precision_score(credoutput_test, predicted_outputcred))
print("Recall value of credit data on test data:", metrics.recall_score(credoutput_test, predicted_outputcred))
print("F1 score of cred data on test data:", metrics.f1_score(credoutput_test, predicted_outputcred))


# In[148]:


print("Accuracy value of credit data on train data:", metrics.accuracy_score(credoutput_train, predicted_outputcred2))
print("Precision value of credit data on train data:", metrics.precision_score(credoutput_train, predicted_outputcred2))
print("Recall value of credit data on train data:", metrics.recall_score(credoutput_train, predicted_outputcred2))
print("F1 score of cred data on train data:", metrics.f1_score(credoutput_train, predicted_outputcred2))


# In[133]:


#tn, fp, fn, tp
print("Confusion matrix of telco data on test data:", metrics.confusion_matrix(output_test, predicted_output))
tn = metrics.confusion_matrix(output_test, predicted_output)[0][0]
fp = metrics.confusion_matrix(output_test, predicted_output)[0][1]
tp = metrics.confusion_matrix(output_test, predicted_output)[1][1]
specificityt = tn / (tn+fp)
fdrt = fp / (fp+tp)
print("Specificity value of telco data on test data:", specificityt)

print("False discovery rate of telco data on test data:", fdrt)


# In[134]:


#tn, fp, fn, tp
print("Confusion matrix of telco data on train data:", metrics.confusion_matrix(output_train, predicted_output2))
tn = metrics.confusion_matrix(output_train, predicted_output2)[0][0]
fp = metrics.confusion_matrix(output_train, predicted_output2)[0][1]
tp = metrics.confusion_matrix(output_train, predicted_output2)[1][1]
specificityt = tn / (tn+fp)
fdrt = fp / (fp+tp)
print("Specificity value of telco data on train data:", specificityt)

print("False discovery rate of telco data on train data:", fdrt)


# In[135]:


print("Confusion matrix of adult data on test data:", metrics.confusion_matrix(adultoutput_test, predicted_outputadult))

tn = metrics.confusion_matrix(adultoutput_test, predicted_outputadult)[0][0]
fp = metrics.confusion_matrix(adultoutput_test, predicted_outputadult)[0][1]
tp = metrics.confusion_matrix(adultoutput_test, predicted_outputadult)[1][1]
specificitya = tn / (tn+fp)
fdra = fp / (fp+tp)
print("Specificity value of adult data on test data:", specificitya)
print("False discovery rate of adult data on test data:", fdra)


# In[136]:


print("Confusion matrix of adult data on train data:", metrics.confusion_matrix(adultoutput_train, predicted_outputadult2))

tn = metrics.confusion_matrix(adultoutput_train, predicted_outputadult2)[0][0]
fp = metrics.confusion_matrix(adultoutput_train, predicted_outputadult2)[0][1]
tp = metrics.confusion_matrix(adultoutput_train, predicted_outputadult2)[1][1]
specificitya = tn / (tn+fp)
fdra = fp / (fp+tp)
print("Specificity value of adult data on test data:", specificitya)
print("False discovery rate of adult data on test data:", fdra)


# In[149]:


print("Confusion matrix of credit data on test data:", metrics.confusion_matrix(credoutput_test, predicted_outputcred))
tn = metrics.confusion_matrix(credoutput_test, predicted_outputcred)[0][0]
fp = metrics.confusion_matrix(credoutput_test, predicted_outputcred)[0][1]
tp = metrics.confusion_matrix(credoutput_test, predicted_outputcred)[1][1]
specificityc = tn / (tn+fp)
fdrc = fp / (fp+tp)
print("Specificity value of credit data on test data:", specificityc)
print("False discovery rate of adult data on test data:", fdrc)


# In[150]:


print("Confusion matrix of credit data on train data:", metrics.confusion_matrix(credoutput_train, predicted_outputcred2))
tn = metrics.confusion_matrix(credoutput_train, predicted_outputcred2)[0][0]
fp = metrics.confusion_matrix(credoutput_train, predicted_outputcred2)[0][1]
tp = metrics.confusion_matrix(credoutput_train, predicted_outputcred2)[1][1]
specificityc = tn / (tn+fp)
fdrc = fp / (fp+tp)
print("Specificity value of credit data on test data:", specificityc)
print("False discovery rate of adult data on test data:", fdrc)


# #Adaboost 

# In[105]:


#Adaboost

def Adaboost(features, output, K):
    m,n = features.shape
    l = output.shape
    
    w1 = [1/n for j in range(n)]
    h = np.zeros((K,n))
    z = np.zeros((K,1))
   
    
    for k in range(K):
        reIndexList = np.random.choice(np.arange(n),n,w1)
        sampleFeature = np.zeros((m,len(reIndexList)))
        sampleOutput =  np.zeros(l)
        for i in range(m):
            for j in range(len(reIndexList)):
                sampleFeature[i][j] = features[i][reIndexList[j]]
                sampleOutput[j] = output[reIndexList[j]]
                
               
        
        w2, b,_ = logistic_regression(sampleFeature, sampleOutput, 0, 300, 0.025)
#         print(w2.shape)
#         h[k] = (np.array(w2)).reshape((len(w2),1))
        h[k] = (np.array(w2)).flatten()
        
        pred_values = predict(features, w2, b)
        error = 0
        
        for j in range(n):
            if pred_values[j] != output[j]:
                error = error + w1[j]
                
        if error>0.5:
            continue
        
        for j in range(n):
            if pred_values[j] == output[j]:
                w1[j] = (w1[j] * error) / (1 - error)
        
        for j in range(len(w1)):
            w1[j] = w1[j] / np.sum(w1)
            
        z[k] = np.log((1-error)/ error)
            
            
    return h, z
            
            
        
        
        
        
        
    


# In[106]:


h, z = Adaboost(features_train, output_train, 5)



# print(h,z)


# In[107]:


hadult, zadult = Adaboost(adultfeatures_train, adultoutput_train, 5)


# In[ ]:


hcred, zcred = Adaboost(credfeatures_train, credoutput_train, 5)


# In[108]:


h10, z10 = Adaboost(features_train, output_train, 10)


# In[109]:


hadult10, zadult10 = Adaboost(adultfeatures_train, adultoutput_train, 10)


# In[ ]:


hcred10, zcred10 = Adaboost(credfeatures_train, credoutput_train, 10)


# In[110]:


h15, z15 = Adaboost(features_train, output_train, 15)


# In[111]:


hadult15, zadult15 = Adaboost(adultfeatures_train, adultoutput_train, 15)


# In[ ]:


hcred15, zcred15 = Adaboost(credfeatures_train, credoutput_train, 15)


# In[112]:


h20, z20 = Adaboost(features_train, output_train, 20)


# In[113]:


hadult20, zadult20 = Adaboost(adultfeatures_train, adultoutput_train, 20)


# In[ ]:


hcred20, zcred20 = Adaboost(credfeatures_train, credoutput_train, 20)


# In[114]:


def predictionAdaboost(feature_test, K, h, z):
    
    
    y_hat = np.zeros((K,len(feature_test)))
    
    for k in range(K):
        y_hat[k] = tanh(np.dot(feature_test, np.transpose(h[k]))) * z[k]
        
    y_hat = np.transpose(y_hat)
    
    avgY_hat = np.zeros((len(feature_test), 1))
    
    for i in range(len(y_hat)):
        value = np.average(y_hat[i])
        
        if value>0 :
            avgY_hat[i] = 1
        else:
            avgY_hat[i] = -1
            
    return avgY_hat
            
        
    


# In[115]:


predicted_adaboost_output5 = predictionAdaboost(features_test, 5, h, z) 
adultpredicted_adaboost_output5 = predictionAdaboost(adultfeatures_test, 5, hadult, zadult) 


print(predicted_adaboost_output5)
print(adultpredicted_adaboost_output5)


# In[137]:


predicted_adaboost_output52 = predictionAdaboost(features_train, 5, h, z) 
adultpredicted_adaboost_output52 = predictionAdaboost(adultfeatures_train, 5, hadult, zadult) 


print(predicted_adaboost_output52)
print(adultpredicted_adaboost_output52)


# In[ ]:


credpredicted_adaboost_output5 = predictionAdaboost(credfeatures_test, 5, hcred, zcred) 
print(credpredicted_adaboost_output5)
credpredicted_adaboost_output52 = predictionAdaboost(credfeatures_train, 5, hcred, zcred) 
print(credpredicted_adaboost_output52)


# In[116]:


predicted_adaboost_output10 = predictionAdaboost(features_test, 10, h10, z10) 
adultpredicted_adaboost_output10 = predictionAdaboost(adultfeatures_test, 10, hadult10, zadult10) 


print(predicted_adaboost_output10)
print(adultpredicted_adaboost_output10)


# In[138]:


predicted_adaboost_output102 = predictionAdaboost(features_train, 10, h10, z10) 
adultpredicted_adaboost_output102 = predictionAdaboost(adultfeatures_train, 10, hadult10, zadult10) 


print(predicted_adaboost_output102)
print(adultpredicted_adaboost_output102)


# In[ ]:


credpredicted_adaboost_output10 = predictionAdaboost(credfeatures_test, 10, hcred10, zcred10) 
print(credpredicted_adaboost_output10)
credpredicted_adaboost_output102 = predictionAdaboost(credfeatures_train, 10, hcred10, zcred10) 
print(credpredicted_adaboost_output102)


# In[117]:


predicted_adaboost_output15 = predictionAdaboost(features_test, 15, h15, z15) 
adultpredicted_adaboost_output15 = predictionAdaboost(adultfeatures_test, 15, hadult15, zadult15) 
 

print(predicted_adaboost_output15)
print(adultpredicted_adaboost_output15)


# In[139]:


predicted_adaboost_output152 = predictionAdaboost(features_train, 15, h15, z15) 
adultpredicted_adaboost_output152 = predictionAdaboost(adultfeatures_train, 15, hadult15, zadult15) 
 

print(predicted_adaboost_output152)
print(adultpredicted_adaboost_output152)


# In[ ]:


credpredicted_adaboost_output15 = predictionAdaboost(credfeatures_test, 15, hcred15, zcred15)
print(credpredicted_adaboost_output15)
credpredicted_adaboost_output152 = predictionAdaboost(credfeatures_train, 15, hcred15, zcred15)
print(credpredicted_adaboost_output152)


# In[118]:


predicted_adaboost_output20 = predictionAdaboost(features_test, 20, h20, z20) 
adultpredicted_adaboost_output20 = predictionAdaboost(adultfeatures_test, 20, hadult20, zadult20) 
 

print(predicted_adaboost_output20)
print(adultpredicted_adaboost_output20)


# In[140]:


predicted_adaboost_output202 = predictionAdaboost(features_train, 20, h20, z20) 
adultpredicted_adaboost_output202 = predictionAdaboost(adultfeatures_train, 20, hadult20, zadult20) 
 

print(predicted_adaboost_output202)
print(adultpredicted_adaboost_output202)


# In[ ]:


credpredicted_adaboost_output20 = predictionAdaboost(credfeatures_test, 20, hcred20, zcred20)
print(credpredicted_adaboost_output20)
credpredicted_adaboost_output202 = predictionAdaboost(credfeatures_train, 20, hcred20, zcred20)
print(credpredicted_adaboost_output202)


# In[119]:


print("Accuracy of adaboost of telco data for k = 5 on test data", metrics.accuracy_score(output_test, predicted_adaboost_output5))
print("Accuracy of adaboost of adult data for k = 5 on test data", metrics.accuracy_score(adultoutput_test, adultpredicted_adaboost_output5))


# In[141]:


print("Accuracy of adaboost of telco data for k = 5 on train data", metrics.accuracy_score(output_train, predicted_adaboost_output52))
print("Accuracy of adaboost of adult data for k = 5 on train data", metrics.accuracy_score(adultoutput_train, adultpredicted_adaboost_output52))


# In[ ]:


print("Accuracy of adaboost of credit data for k = 5 on test data", metrics.accuracy_score(credoutput_test, credpredicted_adaboost_output5))
print("Accuracy of adaboost of credit data for k = 5 on train data", metrics.accuracy_score(credoutput_train, credpredicted_adaboost_output52))


# In[120]:


print("Accuracy of adaboost of telco data for k = 10 on test data", metrics.accuracy_score(output_test, predicted_adaboost_output10))
print("Accuracy of adaboost of adult data for k = 10 on test data", metrics.accuracy_score(adultoutput_test, adultpredicted_adaboost_output10))


# In[142]:


print("Accuracy of adaboost of telco data for k = 10 on train data", metrics.accuracy_score(output_train, predicted_adaboost_output102))
print("Accuracy of adaboost of adult data for k = 10 on train data", metrics.accuracy_score(adultoutput_train, adultpredicted_adaboost_output102))


# In[ ]:


print("Accuracy of adaboost of credit data for k = 10 on test data", metrics.accuracy_score(credoutput_test, credpredicted_adaboost_output10))
print("Accuracy of adaboost of credit data for k = 10 on train data", metrics.accuracy_score(credoutput_train, credpredicted_adaboost_output102))


# In[121]:


print("Accuracy of adaboost of telco data for k = 15 on test data", metrics.accuracy_score(output_test, predicted_adaboost_output15))
print("Accuracy of adaboost of adult data for k = 15 on test data", metrics.accuracy_score(adultoutput_test, adultpredicted_adaboost_output15))


# In[143]:


print("Accuracy of adaboost of telco data for k = 15 on train data", metrics.accuracy_score(output_train, predicted_adaboost_output152))
print("Accuracy of adaboost of adult data for k = 15 on train data", metrics.accuracy_score(adultoutput_train, adultpredicted_adaboost_output152))


# In[ ]:


print("Accuracy of adaboost of credit data for k = 15 on test data", metrics.accuracy_score(credoutput_test, credpredicted_adaboost_output15))
print("Accuracy of adaboost of credit data for k = 15 on train data", metrics.accuracy_score(credoutput_train, credpredicted_adaboost_output152))


# In[122]:


print("Accuracy of adaboost of telco data for k = 20 on test data", metrics.accuracy_score(output_test, predicted_adaboost_output20))
print("Accuracy of adaboost of adult data for k = 20 on test data", metrics.accuracy_score(adultoutput_test, adultpredicted_adaboost_output20))


# In[144]:


print("Accuracy of adaboost of telco data for k = 20 on train data", metrics.accuracy_score(output_train, predicted_adaboost_output202))
print("Accuracy of adaboost of adult data for k = 20 on train data", metrics.accuracy_score(adultoutput_train, adultpredicted_adaboost_output202))


# In[ ]:


print("Accuracy of adaboost of credit data for k = 20 on test data", metrics.accuracy_score(credoutput_test, credpredicted_adaboost_output20))
print("Accuracy of adaboost of credit data for k = 20 on train data", metrics.accuracy_score(credoutput_train, credpredicted_adaboost_output202))


# In[123]:


#tn, fp, fn, tp
print("confusion matrix of adaboost of telco data for k = 5",metrics.confusion_matrix(output_test, predicted_adaboost_output5))
print("confusion matrix of adaboost of adult data for k = 5",metrics.confusion_matrix(adultoutput_test, adultpredicted_adaboost_output5))


# In[ ]:


print("confusion matrix of adaboost of credit data for k = 5",metrics.confusion_matrix(credoutput_test, credpredicted_adaboost_output5))


# In[124]:


#tn, fp, fn, tp
print("confusion matrix of adaboost of telco data for k = 10",metrics.confusion_matrix(output_test, predicted_adaboost_output10))
print("confusion matrix of adaboost of adult data for k = 10",metrics.confusion_matrix(adultoutput_test, adultpredicted_adaboost_output10))


# In[ ]:


print("confusion matrix of adaboost of credit data for k = 10",metrics.confusion_matrix(credoutput_test, credpredicted_adaboost_output10))


# In[125]:


#tn, fp, fn, tp
print("confusion matrix of adaboost of telco data for k = 15",metrics.confusion_matrix(output_test, predicted_adaboost_output15))
print("confusion matrix of adaboost of adult data for k = 15",metrics.confusion_matrix(adultoutput_test, adultpredicted_adaboost_output15))


# In[ ]:


print("confusion matrix of adaboost of credit data for k = 15",metrics.confusion_matrix(credoutput_test, credpredicted_adaboost_output15))


# In[126]:


#tn, fp, fn, tp
print("confusion matrix of adaboost of telco data for k = 20",metrics.confusion_matrix(output_test, predicted_adaboost_output20))
print("confusion matrix of adaboost of adult data for k = 20",metrics.confusion_matrix(adultoutput_test, adultpredicted_adaboost_output20))


# In[ ]:


print("confusion matrix of adaboost of credit data for k = 20",metrics.confusion_matrix(credoutput_test, credpredicted_adaboost_output20))


# In[ ]:




