![image](https://github.com/user-attachments/assets/69cf8e59-ab0e-4513-95bd-9e24d0ef735d)# laptop-pricing
The dataset used for practice labs is a modified subset that maps the price of laptops with different attributes.
The dataset is a filtered and modified version of the Laptop Price Prediction using specifications dataset, available under the Database Contents License (DbCL) v1.0 on the Kaggle website.

### Parameters: The parameters used in the dataset are:

Manufacturer : The company that manufactured the laptop

Category : The category to which the laptop belongs: This parameter is mapped to numerical values in the following way:

Category	Assigned Value
Gaming	1
Netbook	2
Notebook	3
Ultrabook	4
Workstation	5

GPU : The manufacturer of the GPU. This parameter is mapped to numerical values in the following way:

GPU	Assigned Value
AMD	1
Intel	2
NVidia	3
OS
The operating system type (Windows or Linux): This parameter is mapped to numerical values in the following way:

OS	Assigned Value
Windows	1
Linux	2
CPU_core
The type of processor used in the laptop: This parameter is mapped to numerical values in the following way:

CPU_core	Assigned Value
Intel Pentium i3	3
Intel Pentium i5	5
Intel Pentium i7	7
Screen_Size_cm
The size of the laptop screen is recorded in cm.

CPU_frequency
The frequency at which the CPU operates, in GHz.

RAM_GB
The size of the RAM of the system in GB.

Storage_GB_SSD
The size of the SSD storage in GB is installed in the laptop.

Weight_kg
The weight of the laptop is in kgs.

Price
The price of the laptop is in USD.

# Loading data
### Load the dataset to a pandas dataframe named 'df'
```
import pandas as pd
import numpy as np
```
```
file = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"
```
```
df_lt = pd.read_csv(file)
df_lt = head()
```
![image](https://github.com/user-attachments/assets/6f9f5a93-b256-40a9-92f8-b32f924cce95)

print the data types of df columns
```
df_lt.dtypes
```
![image](https://github.com/user-attachments/assets/dfeb382d-2d7e-49fe-8992-55d876595c02)

Print the statistical description of the dataset, including that of 'object' data types.¶
```
df_lt.describe(include ='all')
```
![image](https://github.com/user-attachments/assets/ae7f0fd3-9e82-41fd-a778-bd5a01216001)

Print the summary information of the dataset.
```
df_lt.info()
```
![image](https://github.com/user-attachments/assets/213df84d-253b-4b4c-97c0-e8784281dea6)

## Identify and handle missing values

#### Evaluate the dataset for missing data
```
missing_data = df_lt.isnull()
print(missing_data.head())
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")

```
![image](https://github.com/user-attachments/assets/1499346c-8e43-4ece-a9bd-b10e8c3903a8)

Screen_Size_cm: 40 missing value, Weight_kg : 4 missing value

Replace 'Weight_kg' missing values with mean
```
avg_weight_kg = df_lt['Weight_kg'].astype('float').mean()
df_lt['Weight_kg'] = df_lt['Weight_kg'].fillna(avg_weight_kg)
```
Replace with the most frequent value for Screen_Size_cm
```
df_lt['Screen_Size_cm'].value_counts()
```
![image](https://github.com/user-attachments/assets/098383f5-374a-4bfc-9582-0d6b97ae2927)
39.624 is the most comment size
```
df_lt['Screen_Size_cm']. fillna(39.624)
```

## Data formating
round up the values in Screen_Size_cm into 2 decimal places
```
df_lt['Screen_Size_cm'] = np.round(df[['Screen_Size_cm']],2)
df_lt.head()
```
![image](https://github.com/user-attachments/assets/55de107e-32d0-4099-aeb1-95b69794978f)

change the column 'Unnamed: 0' into No.
```
df_lt.rename(columns = {'Unnamed: 0': 'No'}, inplace = True)
```

## Data Standardization
change the Screen_Size_cm in to inch and Weight_kg to pound
```
df_lt['Screen_Size_inch') = 0.393700787*df_lt['Screen_Size_cm')
df_lt['Weight_pound') = 2.20462262*df_lt['Weight_kg)
```

### Data Normalization
Normalize the "CPU_frequency" attribute with respect to the maximum value available in the dataset.
```
df_lt['CPU_frequency'] = df_lt['CPU_frequency'] /df_lt['CPU_frequency'] .max()
```

### Binning
create 3 bins for the attribute "Price". These bins would be named "Low", "Medium" and "High"
```
bins_lt =np.linspace(min(df_lt['Price']), max(df_lt['Price']),4)
bins_lt
group_names = ['Low', 'Medium', 'High']
df_lt['price_binned'] = pd.cut(df_lt['Price'], bins_lt, labels = group_names, include_lowest = True)
df_lt[['Price', 'price_binned']].head()
```
![image](https://github.com/user-attachments/assets/1587e361-da87-4ba2-a56e-f60e63df5b45)

![image](https://github.com/user-attachments/assets/67eb59dc-13cd-4c8c-a019-6f1cd09c5db8)

 plot the bar graph of these bins.
 ```
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot

plt.pyplot.hist(df_lt["Price"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("Price")
plt.pyplot.ylabel("count")
plt.pyplot.title("Price bins")
```
![image](https://github.com/user-attachments/assets/e6dfc4f1-4364-438e-ab6c-e71bd415d7c2)

### Indicator variables
Convert the "Screen" attribute of the dataset into 2 indicator variables, "Screen-IPS_panel" and "Screen-Full_HD".
```
dummy_variable_1 = pd.get_dummies(df_lt['Screen'])
dummy_variable_1.head()
df_lt = pd.concat([df_lt,dummy_variable_1], axis = 1)
print(df_lt.head())
```
![image](https://github.com/user-attachments/assets/de98bb37-b2cc-4d95-b26a-49d6e8fee8e4)


## save data after clean
```
df_lt.tosave(D:/Data anlysis- working sheet/python/data/laptop_price.csv)
```
# Data Analysis

#### Install Required Libraries
```
import piplite
await piplite.install('seaborn')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
%matplotlib inline
```
import csv file
```
df = pd.read_csv('D:/Data anlysis- working sheet/python/data/laptop_price.csv')
```
## Visualize individual feature patterns

#### calculate the correlation between variables of type "int64" or "float64"
```
numeric_df = df.select_dtypes(include=['int64', 'float64'])
numeric_df.corr(
```
![image](https://github.com/user-attachments/assets/01be9e9b-47ec-483b-aac3-d513a735875c)

```
mask = (corr_matrix < 0.7) & (corr_matrix > -0.7)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, mask=mask)

plt.title("Highlighted Correlation Heatmap", fontsize=15)
plt.show()
```
![image](https://github.com/user-attachments/assets/c0f2c533-8ee9-4dd8-a035-ae83e0720016)

In the heatmap, only RAM_GB (~0.55) has a clear correlation with Price. Other factors may not be prominent or not fully displayed in the image.
For further inspection, I suggest drawing a scatter plot to see the visual relationship between Price and other variables













