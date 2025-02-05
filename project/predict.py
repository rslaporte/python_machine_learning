# %%
import pandas as pd

model = pd.read_pickle('model_rf.pkl')

# %%
df = pd.read_csv('../data/dados_pontos.csv', sep=';')
df

# %%
X = df[model['features']]
predict_proba = model['model'].predict_proba(X)[:,1]

df['prob_active'] = predict_proba
df[["Name", "prob_active"]]