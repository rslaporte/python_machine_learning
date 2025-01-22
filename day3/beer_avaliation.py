# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# %%
df = pd.read_excel("../data/dados_cerveja_nota.xlsx")
df

# %%
plt.plot(df["cerveja"], df["nota"], 'o')
plt.grid(True)
plt.title("Rank x Beer")
plt.ylim(0, 11)
plt.xlim(0, 10)
plt.xlabel("Beer")
plt.ylabel("Rank")
plt.show()


# %%
# Applying the method
reg = linear_model.LinearRegression()
reg.fit(df[["cerveja"]], df["nota"])

# %%
# Calculation the y = a.x + b parameters
a, b = reg.intercept_, reg.coef_[0]

# %%
# Fitting the curve to the points
X = df[["cerveja"]].drop_duplicates()
y = reg.predict(X)
y

plt.plot(df[["cerveja"]], df["nota"],  'o')
plt.plot(X, y,  '-')
plt.grid(True)
plt.title("Rank x Beer")
plt.ylim(0, 11)
plt.xlim(0, 10)
plt.xlabel("Beer")
plt.ylabel("Rank")
plt.show()

# %%
from sklearn import tree
beer_tree = tree.DecisionTreeRegressor(max_depth=2)
beer_tree.fit(df[["cerveja"]], df["nota"])

y_tree = beer_tree.predict(X)
y_tree

plt.plot(df[["cerveja"]], df["nota"],  'o')
plt.plot(X, y,  '-')
plt.plot(X, y_tree,  '-')
plt.grid(True)
plt.title("Rank x Beer")
plt.ylim(0, 11)
plt.xlim(0, 10)
plt.xlabel("Beer")
plt.ylabel("Rank")
plt.legend(["Points", "Linear Regression", "Tree"])
plt.show()