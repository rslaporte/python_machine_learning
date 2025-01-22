# %%
import pandas as pd

# %%
df = pd.read_excel('../data/dados_cerveja_nota.xlsx')
df

# %%
# Creating a binary parameter to classify
df['Aprovado'] = df['nota'] > 5
df

# %%
# Logistic Regression Classification
from sklearn import linear_model
from sklearn import metrics

#Selecting the model
reg = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)

features = ['cerveja']
target = 'Aprovado'

#The model learns
reg.fit(df[features], df[target])

#The model predicts
reg_predict = reg.predict(df[features])

# %%
#Accuracy
reg_acc = metrics.accuracy_score(df[target], reg_predict)
print("Acurácia Reg Log: ", reg_acc)

#Precision
reg_prec = metrics.precision_score(df[target], reg_predict)
print("Precisão Reg Log: ", reg_prec)

#Recall
reg_recall = metrics.recall_score(df[target], reg_predict)
print("Recall Reg Log: ", reg_recall)

#Confusion Metrics
reg_conf = metrics.confusion_matrix(df[target], reg_predict)
reg_conf = pd.DataFrame(reg_conf,
                        index=["False", "True"],
                        columns=["False", "True"])
print(reg_conf)

# %%
#Tree Classification
from sklearn import tree

#Selecting the model
beer_tree = tree.DecisionTreeClassifier(max_depth=2)

#Learning
beer_tree.fit(df[features], df[target])

#Predicting
beer_tree_predict = beer_tree.predict(df[features])

# %%
#Accuracy
tree_acc = metrics.accuracy_score(df[target], beer_tree_predict)
print("Acurácia Árvore: ", tree_acc)

#Precision
tree_precision = metrics.precision_score(df[target], beer_tree_predict)
print("Precisão Árvore: ", tree_precision)

#Recall
tree_recall = metrics.recall_score(df[target], beer_tree_predict)
print("Recall Árvore: ", tree_recall)

tree_conf = metrics.confusion_matrix(df[target], beer_tree_predict)
tree_conf = pd.DataFrame(tree_conf,
                        index=["False", "True"],
                        columns=["False", "True"])
print(tree_conf)

# %%
#Naive Bayes classification
from sklearn import naive_bayes

#Selecting
nb = naive_bayes.GaussianNB()

#Learning
nb.fit(df[features], df[target])

#Predicting
nb_predict = nb.predict(df[features])

# %%
#Accuracy
nb_acc = metrics.accuracy_score(df[target], nb_predict)
print("Acurácia Naive Bayes: ", nb_acc)

#Precision
nb_precision = metrics.precision_score(df[target], nb_predict)
print("Precisão Naive Bayes: ", nb_precision)

#Recall
nb_recall = metrics.recall_score(df[target], nb_predict)
print("Recall Naive Bayes: ", nb_recall)

nb_conf = metrics.confusion_matrix(df[target], nb_predict)
nb_conf = pd.DataFrame(nb_conf,
                       index=['False', 'True'],
                       columns=['False', 'True'])
print(nb_conf)


#Using probas to make a cut
# %%
nb_proba = nb.predict_proba(df[features])[:,1]
nb_predict = nb_proba > 0.2
print(nb_predict)
print(nb_proba)

#Accuracy
nb_acc = metrics.accuracy_score(df[target], nb_predict)
print("Acurácia Naive Bayes: ", nb_acc)

#Precision
nb_precision = metrics.precision_score(df[target], nb_predict)
print("Precisão Naive Bayes: ", nb_precision)

#Recall
nb_recall = metrics.recall_score(df[target], nb_predict)
print("Recall Naive Bayes: ", nb_recall)

nb_conf = metrics.confusion_matrix(df[target], nb_predict)
nb_conf = pd.DataFrame(nb_conf,
                       index=['False', 'True'],
                       columns=['False', 'True'])
print(nb_conf)

# %%
# Analysing the Roc Curve
import matplotlib.pyplot as plt

#Generating Roc curve, which is Sensibility x 1 - Recall
roc_curve = metrics.roc_curve(df[target], nb_proba)
roc_curve

plt.plot(roc_curve[0], roc_curve[1])
plt.grid(True)
plt.plot([0,1], [0,1], '--')
plt.title("Roc Curve")
plt.xlabel('1 - Recall')
plt.ylabel('Sensibility')
plt.show()

#Accuracy, which is the are bellow the curve
roc_auc_score = metrics.roc_auc_score(df[target], nb_proba)
roc_auc_score