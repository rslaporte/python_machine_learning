# %%
import pandas as pd
from sklearn import model_selection

df = pd.read_csv('../data/dados_pontos.csv', sep=';')
df

# %%
#Training and Test bases
features = df.columns[3:-1]
target = 'flActive'

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42)
print("Response Train Tax: ", y_train.mean())
print("Response Test ", y_test.mean())

# %%
#Checking if there're null values
X_train.isna().sum()

# There's 206 null values in the column avgRecorrencia. We will fill these values with
# the max value of the avgRecorrencia column, because we can assume that the null value is from 
# a user that never came back, but it could.
avgRecorrencia_fillna = X_train['avgRecorrencia'].max()
X_train['avgRecorrencia'] = X_train['avgRecorrencia'].fillna(avgRecorrencia_fillna)

#The test base must replicate the values used on test
X_test['avgRecorrencia'] = X_test['avgRecorrencia'].fillna(avgRecorrencia_fillna)

# %%
#Training models: Tree
from sklearn import tree, metrics

model_tree = tree.DecisionTreeClassifier(max_depth=6,
                                         min_samples_leaf=100,
                                         random_state=42)
model_tree.fit(X_train, y_train)

#Accuracy for the Train Base
tree_pred_train = model_tree.predict(X_train)
tree_acc_train = metrics.accuracy_score(y_train, tree_pred_train)
print("Tree Train Acc: ", tree_acc_train)

#Accuracy for the Test Base
tree_pred_test = model_tree.predict(X_test)
tree_acc_test = metrics.accuracy_score(y_test, tree_pred_test)
print("Tree Test Acc: ", tree_acc_test)

#Accuracy for the Train Base
tree_proba_train = model_tree.predict_proba(X_train)[:,1]
tree_auc_train = metrics.roc_auc_score(y_train, tree_proba_train)
print("Tree Train Auc: ", tree_auc_train)

#Accuracy for the Test Base
tree_proba_test = model_tree.predict_proba(X_test)[:,1]
tree_auc_test = metrics.roc_auc_score(y_test, tree_proba_test)
print("Tree Test Auc: ", tree_auc_test)