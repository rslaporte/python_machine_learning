# %%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df_fruit = pd.read_excel("../data/dados_frutas.xlsx")
df_fruit

# %%
#Fruit Tree Example
features = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
target = "Fruta"

X = df_fruit[features]
y = df_fruit[target]

fruit_tree = tree.DecisionTreeClassifier(random_state=42)
fruit_tree.fit(X, y)

plt.figure(dpi=600)

tree.plot_tree(fruit_tree,
                class_names=fruit_tree.classes_,
                feature_names=features,
                filled=True)

#%%
fruit_tree.predict([[0,1,1,1]])[0]

#%%
probs = fruit_tree.predict_proba([[0,1,1,1]])[0]
pd.Series(probs, index=fruit_tree.classes_)

#%%
probs = fruit_tree.predict_proba([[1,1,1,1]])[0]
pd.Series(probs, index=fruit_tree.classes_)
