'''
Ananth Anand P B
VIT Chennai
21BCE1576
'''

import pandas as pd

dict1 = {'Model': ['Mazda RX4', 'Cadillac Fleetwood', 'Chrysler Imperial', 'Honda Civic', 'Dodge Challenger', 'Camaro '
                                                                                                              'Z28',
                   'Pontiac Firebird', 'Porsche 914-2', 'Lotus Europa', 'Maserati Bora'],
         'mpg': [21.0, 10.4, 14.7, 30.4, 15.5, 13.3, 19.2, 26, 30.4, 15],
         'cyl': [6, 8, 8, 4, 8, 8, 8, 4, 4, 8],
         'disp': [160, 472, 440, 75.7, 318, 350, 400, 120.3, 95.1, 301],
         'hp': [110, 205, 230, 52, 150, 245, 175, 91, 113, 335]
         }

df = pd.DataFrame(dict1)

# (1)  printing the dataset
print(df)

# (2)  printing info of dataset
print('\n')
print(df.info())

# (3)  printing descriptive statistics
print('\n')
print(df.describe())

# (4)  Obtaining information about index 4
print('\n')
print(df.loc[4])

# (5)  Checking for any null value in dataset
print('\n')
print(df.isnull().any())
