# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# %%
df = pd.read_parquet("../data/dados_clones.parquet")
df

# %%
# Analysing data
df.groupby(["Status "])[["Estatura(cm)", "Massa(em kilos)"]].mean()

# %%
df["Status_bool"] = df["Status "] == "Apto"
df.groupby(["Distância Ombro a ombro"])["Status_bool"].mean()

# %%
df.groupby(["Tamanho do crânio"])["Status_bool"].mean()

# %%
df.groupby(["Tamanho dos pés"])["Status_bool"].mean()

# %%
df.groupby(["General Jedi encarregado"])["Status_bool"].mean()

# %%
features = [
    "Estatura(cm)",
    "Massa(em kilos)",
    "Distância Ombro a ombro",
    "Tamanho do crânio",
    "Tamanho dos pés",
]

cat_features = [
    "Distância Ombro a ombro",
    "Tamanho do crânio",
    "Tamanho dos pés",
]

# %%
# Encoding the features in 1 or 0 variables
from feature_engine import encoding

X = df[features]
X

onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(X)
X = onehot.transform(X)
X
# %%
clone_tree = tree.DecisionTreeClassifier()
clone_tree.fit(X, df["Status "])

# %%
plt.figure(dpi=600)
tree.plot_tree(clone_tree,
               class_names=clone_tree.classes_,
               feature_names=X.columns,
               filled=True,
               max_depth=3
)