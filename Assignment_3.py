import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

le = LabelEncoder()
data = pd.read_csv(r'C:\Users\ANANTH ANAND P B\Downloads\penguins_size.csv')
print(data.head().to_string())
print(data.info())

''' Uni variate analysis 
'''
sns.distplot(data['culmen_length_mm'])
plt.show()

sns.distplot(data['culmen_depth_mm'])
plt.show()

sns.distplot(data['flipper_length_mm'])
plt.show()

sns.distplot(data['body_mass_g'])
plt.show()

'''Bi Variate analysis
'''

sns.jointplot(x=data['culmen_length_mm'], y=data['culmen_depth_mm'])
plt.show()

'''Multi variate analysis
'''

sns.pairplot(data)
plt.show()


''' 4) Perform descriptive statistics on the dataset.
'''
print(data.describe())

''' 5) Check for missing values and dealing with them
'''
print(data.isnull().any())
print(data.isnull().sum())

'''Replacing null values in numerical columns with median
'''
data['culmen_length_mm'].fillna(data['culmen_length_mm'].median(), inplace=True)
data['culmen_depth_mm'].fillna(data['culmen_depth_mm'].median(), inplace=True)
data['flipper_length_mm'].fillna(data['flipper_length_mm'].median(), inplace=True)
data['body_mass_g'].fillna(data['body_mass_g'].median, inplace=True)

'''Replacing null values in categorical columns with mode
'''
data['sex'].fillna(data['sex'].mode(), inplace=True)

'''6) Finding the outliers and replacing them outliers
'''
# culmen_length_mm
sns.boxplot(data.culmen_length_mm)
plt.show()

# culmen_depth_mm
sns.boxplot(data.culmen_depth_mm)
plt.show()

# flipper_length_mm
sns.boxplot(data.flipper_length_mm)
plt.show()

# # body_mass_g
# sns.boxplot(data.body_mass_g)
# plt.show()

''' 7) Check the correlation of independent variables with the target
'''
print(data.corr().species.sort_values(ascending=False))

''' 8) Check for Categorical columns and perform encoding
'''
data['sex'] = le.fit_transform(data['sex'])
data['species'] = le.fit_transform(data['species'])
data['island'] = le.fit_transform(data['island'])
print(data.head())

''' 9) Split data into independent and dependent variables
'''
x = data.drop(columns=['species'], axis=1)
print(x.head())

y = data['species']
print(y.head())

''' 10) Scaling the data
'''
scale = MinMaxScaler()
x_scaled = pd.DataFrame(scale.fit_transform(x), columns=x.columns)
x_scaled.head()

''' 11) Split the data into training and testing
'''
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)

''' 12) Checking the training and testing data shape
'''
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
