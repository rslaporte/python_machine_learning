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

# %%
#Beer Tree Example
df_beer = pd.read_excel("../data/dados_cerveja.xlsx")
df_beer

# %%
df_beer = df_beer.replace({
    "mud": 1, "pint": 0,
    "sim": 1, "não": 0,
    "clara": 1, "escura": 0,
})

features = ["temperatura", "copo", "espuma", "cor"]
target = "classe"

X = df_beer[features]
y = df_beer[target]

beer_tree = tree.DecisionTreeClassifier(random_state=42)
beer_tree.fit(X, y)

plt.figure(dpi=600)

tree.plot_tree(beer_tree,
            class_names=beer_tree.classes_,
            feature_names=features,
            filled=True)

probs = beer_tree.predict_proba([[1,1,1,1]])[0]
pd.Series(probs, index=beer_tree.classes_)
