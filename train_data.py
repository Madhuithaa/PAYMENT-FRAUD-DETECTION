# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:10:49 2022

@author: okokp
"""

from sklearn.preprocessing import RobustScaler #robust normlization for outliers
import sklearn.metrics as metrics #metrics librry
import seaborn as sns # for intractve graphs
from sklearn.ensemble import RandomForestClassifier #Random Forest
import matplotlib.pyplot as plt #for visualization
from xgboost.sklearn import XGBClassifier #XGBoost
from sklearn.metrics import classification_report
from xgboost import plot_importance #feature rimportance
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model, load_model
from keras.layers import Input,Dense, BatchNormalization #layers of autoencoder
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping #callbacks
from keras import regularizers #regularization
from sklearn.ensemble import IsolationForest #Isolation Forest
from sklearn.mixture import GaussianMixture #Gaussian Mixture
from imblearn.over_sampling import SMOTE
import os

dataframe = pd.read_csv("creditcard.csv")

RS=RobustScaler()
dataframe['Amount'] = RS.fit_transform(dataframe['Amount'].values.reshape(-1, 1))
dataframe['Time'] = RS.fit_transform(dataframe['Time'].values.reshape(-1, 1))
df = dataframe.sample(frac=1, random_state = 42)
df.head()

print('Normal', round(
        df['Class'].value_counts()[0]/len(df)*100, 2), '% of the dataset')
print('Fraud', round(
        df['Class'].value_counts()[1]/len(df)*100, 2), '% of the dataset')
sns.countplot("Class",data=df)
fraud_df_train = df.loc[df['Class'] == 1][:int(492*0.8)]
fraud_df_test = df.loc[df['Class'] == 1][int(492*0.8):]


#undersampling of the data. Fraude represent 10% of base now
normal_df_train_sup= df.loc[df['Class'] == 0][:int(492*0.8*9*3)]
normal_df_test= df.loc[df['Class'] == 0][int(492*0.8)*9*3:int(492*0.8*9*3)+int(284800*0.2)]
new_df_train = pd.concat([pd.DataFrame(normal_df_train_sup), fraud_df_train])

#oversampling of the data. The number of Fraud was twiced
sm = SMOTE(k_neighbors=5, random_state=0, n_jobs=8)

normal_df_train_sup, fraud_df_train = sm.fit_resample(new_df_train.drop('Class', axis=1), new_df_train['Class'])
fraud_df_train = pd.DataFrame(fraud_df_train.transpose()).rename(columns={0:"Class"})
new_df_train = pd.concat([pd.DataFrame(normal_df_train_sup), fraud_df_train ], axis=1)
new_df_train.columns=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount','Class']
new_df_test = pd.concat([pd.DataFrame(normal_df_test), fraud_df_test])

new_df_train

print('Normal', round(
        new_df_test['Class'].value_counts()[0]/len(new_df_test)*100, 2), '% of the test dataset')
print('Fraud', round(
        new_df_test['Class'].value_counts()[1]/len(new_df_test)*100, 2), '% of the test dataset')
sns.countplot("Class",data=new_df_test).set_title('Class Count - Test Data')

X_train_sup = new_df_train.drop('Class', axis=1)
y_train = new_df_train['Class']

X_test=new_df_test.drop('Class', axis=1)
y_test=new_df_test['Class']
new_df_train.head()

print('Normal', round(
        pd.Series(y_train).value_counts()[0]/len(X_train_sup)*100, 2), '% of the train dataset')
print('Fraude', round(
        pd.Series(y_train).value_counts()[1]/len(X_train_sup)*100, 2), '% of the train dataset')
sns.countplot("Class",data=new_df_train).set_title('Class Count - Train Data')

print("Training")

#grid_search
"""
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, 
                               n_iter = 100, verbose=2, n_jobs = -1)
rf_random.fit(X_res,y_res)
rf_random.best_params_
"""

rfc = RandomForestClassifier(n_estimators = 1600,
 min_samples_split = 2,
 min_samples_leaf = 1,
 max_features = 'sqrt',
 max_depth = 100,
 bootstrap = False);

xgb = XGBClassifier(min_child_weight = 5,
 max_depth=12,
 learning_rate= 0.1,
 gamma= 0.2,
 colsample_bytree= 0.7)

rfc.fit(X_train_sup,y_train)
prediçtion_rfc = rfc.predict_proba(X_test.values)
tresholds = np.linspace(0 , 1 , 200)
scores_rfc=[]
for treshold in tresholds:
    y_hat_rfc = (prediçtion_rfc[:,0] < treshold).astype(int)
    scores_rfc.append([metrics.recall_score(y_pred=y_hat_rfc, y_true=y_test),
                 metrics.precision_score(y_pred=y_hat_rfc, y_true=y_test),
                 metrics.fbeta_score(y_pred=y_hat_rfc, y_true=y_test, beta=2),
                 metrics.cohen_kappa_score(y1=y_hat_rfc, y2=y_test)])
scores_rfc = np.array(scores_rfc)
final_tresh = tresholds[scores_rfc[:, 2].argmax()]
y_hat_rfc = (prediçtion_rfc < final_tresh).astype(int)
cm_rfc = metrics.confusion_matrix(y_test,y_hat_rfc[:,0])
best_score = scores_rfc[scores_rfc[:, 2].argmax(),:]
recall_score = best_score[0]
precision_score = best_score[1]
fbeta_score = best_score[2]
cohen_kappa_score = best_score[3]

print('The recall score is: %.3f' % recall_score)
print('The precision score is: %.3f' % precision_score)
print('The f2 score is: %.3f' % fbeta_score)
print('The Kappa score is: %.3f' % cohen_kappa_score)


predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount']
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': rfc.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance - Random Forest',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()  


y_true = y_test
y_pred_rfc = y_hat_rfc[:,0]
target_names = ['class 0 (Normal)', 'class 1 (Fraud)']
print(classification_report(y_true, y_pred_rfc, target_names=target_names))

y_true = y_test
y_pred_rfc = y_hat_rfc[:,0]
target_names = ['class 0 (Normal)', 'class 1 (Fraud)']
print(classification_report(y_true, y_pred_rfc, target_names=target_names))


#precision_recall
plt.plot(tresholds, scores_rfc[:, 0], label='$Recall$')
plt.plot(tresholds, scores_rfc[:, 1], label='$Precision$')
plt.plot(tresholds, scores_rfc[:, 2], label='$F_2$')
plt.ylabel('Score')
# plt.xticks(np.logspace(-10, -200, 3))
plt.xlabel('Threshold')
plt.legend(loc='best')
plt.title('trade off Precision - Recall for Random Forrest')
plt.show()

xgb = XGBClassifier(min_child_weight = 5,
 max_depth=12,
 learning_rate= 0.1,
 gamma= 0.2,
 colsample_bytree= 0.7)

xgb.fit(X_train_sup, y_train)

prediction_xgb = xgb.predict_proba(X_test)
tresholds = np.linspace(0 , 1 , 200)
scores_xgb=[]
for treshold in tresholds:
    y_hat_xgb = (prediction_xgb[:,0] < treshold).astype(int)
    scores_xgb.append([metrics.recall_score(y_pred=y_hat_xgb, y_true=y_test),
                 metrics.precision_score(y_pred=y_hat_xgb, y_true=y_test),
                 metrics.fbeta_score(y_pred=y_hat_xgb, y_true=y_test, beta=2),
                 metrics.cohen_kappa_score(y1=y_hat_xgb, y2=y_test)])  
scores_xgb = np.array(scores_xgb)
final_tresh = tresholds[scores_xgb[:, 2].argmax()]
y_hat_xgb = (prediction_xgb < final_tresh).astype(int)
best_score_xgb = scores_xgb[scores_xgb[:, 2].argmax(),:]
recall_score_xgb = best_score_xgb[0]
precision_score_xgb = best_score_xgb[1]
fbeta_score_xgb = best_score_xgb[2]
cohen_kappa_score_xgb = best_score_xgb[3]

print('The recall score is": %.3f' % recall_score_xgb)
print('The precision score is": %.3f' % precision_score_xgb)
print('The f2 score is": %.3f' % fbeta_score_xgb)
print('The Kappa score is": %.3f' % cohen_kappa_score_xgb)


y_true = y_test
y_pred_xgb = y_hat_xgb[:,0]
target_names = ['class 0 (Normal)', 'class 1 (Fraud)']
print(classification_report(y_true, y_pred_xgb, target_names=target_names))
cm = pd.crosstab(y_test, y_pred_xgb, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix for XGBoost', fontsize=14)
plt.show()
fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
plot_importance(xgb, height=0.8, title="Features importance - XGBoost", ax=ax, color="blue") 
plt.show()


#precision_recall
plt.plot(tresholds, scores_xgb[:, 0], label='$Recall$')
plt.plot(tresholds, scores_xgb[:, 1], label='$Precision$')
plt.plot(tresholds, scores_xgb[:, 2], label='$F_2$')
plt.ylabel('Score')
# plt.xticks(np.logspace(-10, -200, 3))
plt.xlabel('Threshold')
plt.legend(loc='best')
plt.title('trade off Precision - Recall for XGBoost')
plt.show()

scores=[]
for i in  list(range(1,11)):
    df = dataframe.sample(frac=1, random_state = i)
    #divid frauds in train (80%) and test (20%)
    fraud_df_train = df.loc[df['Class'] == 1][:int(492*0.8)]
    fraud_df_test = df.loc[df['Class'] == 1][int(492*0.8):]

    # undersmpling of norml data: the normal data represent 90% of the new base
    normal_df_train_sup= df.loc[df['Class'] == 0][:int(492*0.8*9)]
    normal_df_test= df.loc[df['Class'] == 0][int(492*0.8)*9:int(492*0.8*9)+int(284807*0.2)]
    new_df_train = pd.concat([normal_df_train_sup, fraud_df_train])
    new_df_test = pd.concat([normal_df_test, fraud_df_test])

    X_train_sup = new_df_train.drop('Class', axis=1)
    y_train = new_df_train['Class']
    X_test=new_df_test.drop('Class', axis=1)
    y_test=new_df_test['Class']

    rfc.fit(X_train_sup,y_train)
    prediction_rfc = rfc.predict_proba(X_test.values)
    tresholds = np.linspace(0 , 1 , 200)
    scores_rfc=[]
    for treshold in tresholds:
        y_hat_rfc = (prediction_rfc[:,0] < treshold).astype(int)
        scores_rfc.append([metrics.recall_score(y_pred=y_hat_rfc, y_true=y_test),
                 metrics.precision_score(y_pred=y_hat_rfc, y_true=y_test),
                 metrics.fbeta_score(y_pred=y_hat_rfc, y_true=y_test, beta=2),
                 metrics.cohen_kappa_score(y1=y_hat_rfc, y2=y_test)])
    scores_rfc = np.array(scores_rfc)
    #choice the model with best f2 score
    best_scores = scores_rfc[scores_rfc[:, 2].argmax(),:]
    scores.append(best_scores)
    print('recall, precision, f2, kappa in shuffle %1d' %i)
    print(best_scores)

    recall_score_rfc = np.mean(scores, axis=0)[0]
    precision_score_rfc = np.mean(scores,axis=0)[1]
    fbeta_score_rfc = np.mean(scores, axis=0)[2]
    cohen_kappa_score_rfc = np.mean(scores, axis=0)[3]
print('--------------------------------------------------')
print("for the random forest algorithm:")
print('The recall score is: %.3f' % recall_score_rfc)
print('The precision score is: %.3f' % precision_score_rfc)
print('The f2 score is: %.3f' % fbeta_score_rfc)
print('The Kappa score is: %.3f' % cohen_kappa_score_rfc)


scores=[]

for i in  list(range(1,11)):
    df = dataframe.sample(frac=1, random_state = i)
    #divid frauds in train (80%) and test (20%)
    fraud_df_train = df.loc[df['Class'] == 1][:int(492*0.8)]
    fraud_df_test = df.loc[df['Class'] == 1][int(492*0.8):]

    # undersmpling of norml data: the normal data represent 90% of the new base
    normal_df_train_sup= df.loc[df['Class'] == 0][:int(492*0.8*9)]
    normal_df_test= df.loc[df['Class'] == 0][int(492*0.8)*9:int(492*0.8*9)+int(284807*0.2)]
    new_df_train = pd.concat([normal_df_train_sup, fraud_df_train])
    new_df_test = pd.concat([normal_df_test, fraud_df_test])
    
    X_train_sup = new_df_train.drop('Class', axis=1)
    y_train = new_df_train['Class']
    X_test=new_df_test.drop('Class', axis=1)
    y_test=new_df_test['Class']

    xgb.fit(X_train_sup.values,y_train)
    prediction_xgb = xgb.predict_proba(X_test.values)
    tresholds = np.linspace(0 , 1 , 200)
    scores_xgb=[]
    for treshold in tresholds:
        y_hat_xgb = (prediction_xgb[:,0] < treshold).astype(int)
        scores_xgb.append([metrics.recall_score(y_pred=y_hat_xgb, y_true=y_test),
                 metrics.precision_score(y_pred=y_hat_xgb, y_true=y_test),
                 metrics.fbeta_score(y_pred=y_hat_xgb, y_true=y_test, beta=2),
                 metrics.cohen_kappa_score(y1=y_hat_xgb, y2=y_test)])
    scores_xgb = np.array(scores_xgb)
    #choice the model with best f2 score
    best_scores = scores_xgb[scores_xgb[:, 2].argmax(),:]
    
    print('recall, precision, f2, kappa in shuffle %1d' %i)
    print(best_scores)
    scores.append(best_scores)
    
recall_score_xgb = np.mean(scores, axis=0)[0]
precision_score_xgb = np.mean(scores,axis=0)[1]
fbeta_score_xgb = np.mean(scores, axis=0)[2]
cohen_kappa_score_xgb = np.mean(scores, axis=0)[3]

print('---------------------------------------------')
print("for the XGBoost algorithm classifier:")
print('The recall score is: %.3f' % recall_score_xgb)
print('The precision score is: %.3f' % precision_score_xgb)
print('The f2 score is: %.3f' % fbeta_score_xgb)
print('The Kappa score is: %.3f' % cohen_kappa_score_xgb)

i = 0
t0 = dataframe.loc[df['Class'] == 0]
t1 = dataframe.loc[df['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in predictors:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


print("Preparation of Data")

df = dataframe.sample(frac=1, random_state=42)
fraude_df_train = df.loc[df['Class'] == 1][:int(492*0.8)]
fraude_df_test = df.loc[df['Class'] == 1][int(492*0.8):]

normal_df_test= df.loc[df['Class'] == 0][int(492*0.8)*9:int(492*0.8*9)+int(284807*0.2)]

new_df_train_semisup= df.loc[df['Class'] == 0][:int(284807*.8)]
new_df_test = pd.concat([normal_df_test, fraude_df_test])

X_train_semisup = new_df_train_semisup.drop('Class', axis=1)
#X_train_semisup=X_train_semisup[['V4','V5', 'V7','V9','V10','V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'Amount']]
X_test = new_df_test.drop('Class', axis=1)
#X_test=X_test[['V4','V5', 'V7','V9','V10','V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'Amount']]
y_test = new_df_test['Class']

X_train_semisup.head()

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, n_init=5)
gmm.fit(X_train_semisup)

prediction_MG = gmm.score_samples(X_test)

scores_MG = []
tresholds = np.linspace(-1000 , 100 , 200)

for treshold in tresholds:
    y_hat_MG = (prediction_MG < treshold).astype(int)
    scores_MG.append([metrics.recall_score(y_pred=y_hat_MG, y_true=y_test),
                 metrics.precision_score(y_pred=y_hat_MG, y_true=y_test),
                 metrics.fbeta_score(y_pred=y_hat_MG, y_true=y_test, beta=2),
                 metrics.cohen_kappa_score(y1=y_hat_MG, y2=y_test)])
    
scores_MG=np.array(scores_MG)
final_tresh = tresholds[scores_MG[:, 2].argmax()]
y_hat_MG = (prediction_MG < final_tresh).astype(int)
best_score = scores_MG[scores_MG[:, 2].argmax(),:]
recall_score = best_score[0]
precision_score = best_score[1]
fbeta_score = best_score[2]
cohen_kappa_score = best_score[3]

model_IF = IsolationForest(random_state=42, n_jobs=4, max_samples=X_train_semisup.shape[0], bootstrap=False, n_estimators=100)
model_IF.fit(X_train_semisup)
tresholds = np.linspace(-.2, .2, 200)
prediction_IF = model_IF.decision_function(X_test)
scores_IF = []
for treshold in tresholds:
    y_hat_IF = (prediction_IF < treshold).astype(int)
    scores_IF.append([metrics.recall_score(y_pred=y_hat_IF, y_true=y_test),
                 metrics.precision_score(y_pred=y_hat_IF, y_true=y_test),
                 metrics.fbeta_score(y_pred=y_hat_IF, y_true=y_test, beta=2),
                 metrics.cohen_kappa_score(y1=y_hat_IF, y2=y_test)])
scores_IF=np.array(scores_IF)
a=scores_IF[:,2].argmax()
opt_trh=tresholds[a]
y_hat_IF = (prediction_IF < opt_trh).astype(int)
best_score = scores_IF[scores_IF[:, 2].argmax(),:]
recall_score = best_score[0]
precision_score = best_score[1]
fbeta_score = best_score[2]
cohen_kappa_score = best_score[3]

print('The recall score is": %.3f' % recall_score)
print('The precision score is": %.3f' % precision_score)
print('The f2 score is": %.3f' % fbeta_score)
print('The Kappa score is": %.3f' % cohen_kappa_score)
target_names = ['class 0 (Normal)', 'class 1 (Fraud)']
print(classification_report(y_test, y_hat_IF, target_names=target_names))

cm = pd.crosstab(y_test, y_hat_IF, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix for Gaussian Mix', fontsize=14)
plt.show()

print('The recall score is": %.3f' % recall_score)
print('The precision score is": %.3f' % precision_score)
print('The f2 score is": %.3f' % fbeta_score)
print('The Kappa score is": %.3f' % cohen_kappa_score)
target_names = ['class 0 (Normal)', 'class 1 (Fraud)']
print(classification_report(y_test, y_hat_MG, target_names=target_names))

cm = pd.crosstab(y_test, y_hat_MG, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix for Gaussian Mix', fontsize=14)
plt.show()

from numpy.random import seed
seed(4)
input_dim = X_train_semisup.shape[1]

Input_layer=Input(shape=(input_dim,))
#encoder
encoding_dim = 100
from keras import initializers
#init = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=42)
#encoder
encoder=Dense(units=encoding_dim,activation='tanh', use_bias=False, 
              activity_regularizer=regularizers.l1(10e-5))(Input_layer)
encoder=BatchNormalization()(encoder)
encoder=Dense(units=int(encoding_dim/2), activation='relu')(encoder)
encoder=Dense(units=int(encoding_dim/4), activation='relu')(encoder)

#decoder
decoder=Dense(units=int(encoding_dim/2), activation='relu')(encoder)
decoder=Dense(units=int(encoding_dim), activation='relu')(decoder)
decoder=Dense(units=input_dim,activation='relu')(decoder)


#modelo
autoencoder=Model(inputs=Input_layer,outputs=decoder)
encodermodel=Model(inputs=Input_layer,outputs=encoder)

epochs= 400
batch_size=128

#TensorBoard=TensorBoard(log_dir='./logs',histogram_freq=0,write_grads=True,write_images=True)
best_weights_filepath = './best_weights.hdf5'

earlyStopping=EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')

saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', 
                                verbose=1, save_best_only=True, mode='auto')
#we use 90% of data for train and 10% for validation
autoencoder.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

history=autoencoder.fit(X_train_semisup[5000*0:5000*1],X_train_semisup[5000*0:5000*1],
                    epochs=epochs,batch_size=batch_size,shuffle=False,
                    validation_split=0.1,verbose=0, 
                    callbacks=[saveBestModel,earlyStopping]).history
autoencoder.load_weights(best_weights_filepath)
predictions_AE=autoencoder.predict(X_test)
#mse is the mean squared error between the original data points and the reconstruction data points
mse=np.mean(np.power(X_test - predictions_AE, 2), axis=1)

#plotting the history of the model\qz
plt.figure()
plt.plot(history['loss'],label='loss')
plt.legend()
plt.plot(history['val_loss'],label='validation loss')
plt.legend()
plt.title('loss in train and validation split')

fpr, tpr, tresholds = metrics.roc_curve(y_test, mse)
scores_AE=[]
for treshold in tresholds:
    y_hat_AE = (mse > treshold).astype(int)
    scores_AE.append([metrics.recall_score(y_pred=y_hat_AE, y_true=y_test),
                 metrics.precision_score(y_pred=y_hat_AE, y_true=y_test),
                 metrics.fbeta_score(y_pred=y_hat_AE, y_true=y_test, beta=2),
                 metrics.cohen_kappa_score(y1=y_hat_AE, y2=y_test)])

scores_AE = np.array(scores_AE)
final_tresh = tresholds[scores_AE[:, 2].argmax()]
y_hat_AE = (mse > final_tresh).astype(int)
best_score_AE = scores_AE[scores_AE[:, 2].argmax(),:]
recall_score_AE = best_score_AE[0]
precision_score_AE = best_score_AE[1]
fbeta_score_AE = best_score_AE[2]
cohen_kappa_score_AE = best_score_AE[3]

print('The recall score is": %.3f' % recall_score_AE)
print('The precision score is": %.3f' % precision_score_AE)
print('The f2 score is": %.3f' % fbeta_score_AE)
print('The Kappa score is": %.3f' % cohen_kappa_score_AE)
target_names = ['class 0 (Normal)', 'class 1 (Fraud)']
print(classification_report(y_test, y_hat_AE, target_names=target_names))

cm = pd.crosstab(y_test, y_hat_AE, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix for Autoencoder', fontsize=14)
plt.show()

y_test=np.array(y_test)
mse=np.array(mse)
error_df=pd.DataFrame({'reconstruction_error':mse,'true_class':y_test})
t0 = error_df.loc[error_df['true_class'] == 0]
t1 = error_df.loc[error_df['true_class'] == 1]
t2=final_tresh*np.ones((60000,), dtype=int)
sns.scatterplot(data = t0['reconstruction_error'] ,label="Normal")
sns.lineplot(data=t2, color='black', label='treshold')
sns.scatterplot(data = t1['reconstruction_error'],label="Fraud").set_title('Reconstruction error')