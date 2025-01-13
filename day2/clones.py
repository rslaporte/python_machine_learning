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
    "General Jedi encarregado"
]

X = df[features]
X

# %%
# Encoding the features in 1 or 0 variables
from feature_engine import encoding
onehot = encoding.OneHotEncoder(X)