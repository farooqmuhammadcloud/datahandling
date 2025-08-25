import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

# -----------------------------
# 1. Simulate multivariate time series
# -----------------------------
np.random.seed(42)
n_obs = 200

# Two latent factors following AR(1) processes
f1 = np.zeros(n_obs)
f2 = np.zeros(n_obs)
for t in range(1, n_obs):
    f1[t] = 0.8 * f1[t-1] + np.random.normal()
    f2[t] = 0.5 * f2[t-1] + np.random.normal()

# Factor loadings (how factors influence observed series)
Lambda = np.array([[1.0, 0.5],
                   [0.7, 0.3],
                   [0.2, 1.0]])

# Observed series = Lambda * factors + noise
errors = np.random.normal(scale=0.3, size=(n_obs, 3))
Y = np.dot(np.column_stack([f1, f2]), Lambda.T) + errors

data = pd.DataFrame(Y, columns=['y1', 'y2', 'y3'])

# -----------------------------
# 2. Fit Dynamic Factor Model (MLFA)
# -----------------------------
model = DynamicFactor(data, k_factors=2, factor_order=1)
result = model.fit(maxiter=1000, disp=False)

print(result.summary())
#result.model.k_factors

# -----------------------------
# 3. Plot estimated latent factors (convert to DataFrame first)
# -----------------------------
# Transpose the factors array to match the expected shape (200, 2)
factors = pd.DataFrame(
    result.factors.filtered.T,  # Added .T to transpose the array
    index=data.index,
    columns=[f"Factor{i+1}" for i in range(result.model.k_factors)]
)

factors.plot(subplots=True, figsize=(10, 6), title="Estimated Latent Factors")
plt.show()
