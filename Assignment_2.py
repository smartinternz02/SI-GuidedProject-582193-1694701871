import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

df = pd.read_csv(r'C:\Users\ANANTH ANAND P B\Downloads\archive\House Price India.csv')

print(df.head().to_string())

# univariate analysis
sns.distplot(df['lot area'])
plt.show()


# bivariate analysis
sns.jointplot(x='living area', y='lot area', data=df)
plt.show()


# # mutlivariate analysis
# sns.pairplot(df)
# plt.show()

print(df.describe().to_string())
print(df.isnull().any())

