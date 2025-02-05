# %%
#Importing table
import pandas as pd
from sklearn import model_selection

df = pd.read_csv("../data/dados_pontos.csv", sep=';')
# %%
features = df.columns.tolist()[3:-1]
target = 'flActive'

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features], 
                                                                    df[target],
                                                                    test_size = 0.2,
                                                                    random_state = 42, 
                                                                    stratify=df[target]
                                                                )

X_train

# %%
print("Tx Answer Train:" , y_train.mean())
print("Tx Answer Test", y_test.mean())

# %%
from sklearn import pipeline
from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import pipeline
from sklearn import metrics
import scikitplot as skplt

from feature_engine import imputation

# %%
#Creating a pipepline to perform the necessary transformations
X_test.isna().sum().T #Variables that are null
max_avgRecorrencia = X_test['avgRecorrencia'].max()

#Inputing the null values of avgRecorrencia
imputation_max = imputation.ArbitraryNumberImputer(variables=['avgRecorrencia'],
                                                   arbitrary_number=max_avgRecorrencia)

#Inputing 0s
features_imput_0 = [
    'qtdeRecencia',
    'freqDias',
    'freqTransacoes',
    'qtdListaPresença',
    'qtdChatMessage',
    'qtdTrocaPontos',
    'qtdResgatarPonei',
    'qtdPresençaStreak',
    'pctListaPresença',
    'pctChatMessage',
    'pctTrocaPontos',
    'pctResgatarPonei',
    'pctPresençaStreak',
    'qtdePontosGanhos',
    'qtdePontosGastos',
    'qtdePontosSaldo',
]

imputation_0 = imputation.ArbitraryNumberImputer(variables=features_imput_0,
                                                 arbitrary_number=0)

# %%

#Defining the model
#Tree
model_tree = tree.DecisionTreeClassifier(max_depth=4, 
                                         min_samples_leaf=50,
                                         random_state=42)

#Random Forest Classifier
model_forest = ensemble.RandomForestClassifier(random_state=42)

#Using grid to select the optimal parameters for the model
params = {
    "n_estimators": [100,150,200,250,500],
    "min_samples_leaf": [10,20,30,50,100]
}

grid = model_selection.GridSearchCV(model_forest,
                                    param_grid=params,
                                    n_jobs=-1,
                                    scoring='roc_auc')

#Creating a pipeline
my_pipeline = pipeline.Pipeline([
    ('imput_0', imputation_0), #Inputs 0 on all variables, except avgRecorrencia
    ('imput_max', imputation_max), #Inputs the max on avgRecorrencia
    ('model', grid) #Create the model, (a tree in this case)
])

#%%
#Training the model
my_pipeline.fit(X_train, y_train)

# %%
grid.best_params_

#%%
#Testing the model
y_train_predict = my_pipeline.predict(X_train)
y_train_proba = my_pipeline.predict_proba(X_train)[:,1]

y_test_predict = my_pipeline.predict(X_test)
y_test_proba = my_pipeline.predict_proba(X_test)

#%%
#Metrics of performance
acc_test = metrics.accuracy_score(y_test, y_test_predict)
acc_train = metrics.accuracy_score(y_train, y_train_predict)
print("Accuracy Score Test: ", acc_test)
print("Accuracy Score Train: ", acc_train)

auc_test = metrics.roc_auc_score(y_test, y_test_proba[:,1])
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
print("Roc Score Test: ", auc_test)
print("Roc Score Train: ", auc_train)

#%%
#Determining the importance of each variable in the model
f_importance = my_pipeline[-1].best_estimator_.feature_importances_
pd.Series(f_importance, index=features).sort_values(ascending=False)

#%%
#Ploting the ROC Curve
import matplotlib.pyplot as plt
plt.figure(dpi=600)
skplt.metrics.plot_roc(y_test, y_test_proba)
plt.show()

#%%
#Cumulative gain seems like recall metric
skplt.metrics.plot_cumulative_gain(y_test, y_test_proba)

#%%
#Example
user_test = pd.DataFrame({
    "TRUE" : y_test,
    "proba" : y_test_proba[:,1]
})

user_test = user_test.sort_values("TRUE", ascending=False) #Sorting
user_test['cum_true'] = user_test["TRUE"].cumsum() #Cumulative sum
user_test['capture_rate'] = user_test['cum_true'] / user_test['TRUE'].sum() #Catpure rate. It expects to catpure much with little sample percentage
user_test

# %%
#Lift Curve
skplt.metrics.plot_lift_curve(y_test, y_test_proba)

# %%
#KS Curve
skplt.metrics.plot_ks_statistic (y_test, y_test_proba)

# %%
#Saving the model
model_s = pd.Series({
    "model": my_pipeline,
    "features": features,
    "auc_test": auc_test
})

model_s.to_pickle("model_rf.pkl")