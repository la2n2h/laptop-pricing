# laptop-pricing
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

# DATA PREPARE
```
# import libary packages
import pandas as pd
import numpy as np

# read file
file = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"
df_lt = pd.read_csv(file)
```
# DATA PROCESS
### data understanding
```
# check first few rows
df_lt = head()
```
![image](https://github.com/user-attachments/assets/6f9f5a93-b256-40a9-92f8-b32f924cce95)

```
# print the data types of df columns
df_lt.dtypes
```
![image](https://github.com/user-attachments/assets/dfeb382d-2d7e-49fe-8992-55d876595c02)

```
# Print the statistical description of the dataset, including that of 'object' data types
df_lt.describe(include ='all')
```
![image](https://github.com/user-attachments/assets/ae7f0fd3-9e82-41fd-a778-bd5a01216001)

```
# Print the summary information of the dataset.
df_lt.info()
```
![image](https://github.com/user-attachments/assets/213df84d-253b-4b4c-97c0-e8784281dea6)

### Identify and handle missing values

```
# Evaluate the dataset for missing data
missing_data = df_lt.isnull()
print(missing_data.head())
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
```
![image](https://github.com/user-attachments/assets/1499346c-8e43-4ece-a9bd-b10e8c3903a8)

Screen_Size_cm: 40 missing value, Weight_kg : 4 missing value
```
# Replace 'Weight_kg' missing values with mean
avg_weight_kg = df_lt['Weight_kg'].astype('float').mean()
df_lt['Weight_kg'] = df_lt['Weight_kg'].fillna(avg_weight_kg)
```
```
# Replace with the most frequent value for Screen_Size_cm
df_lt['Screen_Size_cm'].value_counts()
```
![image](https://github.com/user-attachments/assets/098383f5-374a-4bfc-9582-0d6b97ae2927)
39.624 is the most comment size
```
df_lt['Screen_Size_cm']. fillna(39.624)
```

### Data formating
```
# round up the values in Screen_Size_cm into 2 decimal places
df_lt['Screen_Size_cm'] = np.round(df[['Screen_Size_cm']],2)
df_lt.head()
```
![image](https://github.com/user-attachments/assets/55de107e-32d0-4099-aeb1-95b69794978f)

```
# change the column 'Unnamed: 0' into No.
df_lt.rename(columns = {'Unnamed: 0': 'No'}, inplace = True)
```
### Data Standardization

```
#change the Screen_Size_cm in to inch and Weight_kg to pound
df_lt['Screen_Size_inch') = 0.393700787*df_lt['Screen_Size_cm')
df_lt['Weight_pound') = 2.20462262*df_lt['Weight_kg)
```
### Data Normalization
```
#Normalize the "CPU_frequency" attribute with respect to the maximum value available in the dataset.
df_lt['CPU_frequency'] = df_lt['CPU_frequency'] /df_lt['CPU_frequency'] .max()
```

### Binning

```
# create 3 bins for the attribute "Price". These bins would be named "Low", "Medium" and "High"
bins_lt =np.linspace(min(df_lt['Price']), max(df_lt['Price']),4)
bins_lt
group_names = ['Low', 'Medium', 'High']
df_lt['price_binned'] = pd.cut(df_lt['Price'], bins_lt, labels = group_names, include_lowest = True)
df_lt[['Price', 'price_binned']].head()
```
![image](https://github.com/user-attachments/assets/1587e361-da87-4ba2-a56e-f60e63df5b45)

![image](https://github.com/user-attachments/assets/67eb59dc-13cd-4c8c-a019-6f1cd09c5db8)

 ```
# plot the bar graph of these bins.
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

```
# Convert the "Screen" attribute of the dataset into 2 indicator variables, "Screen-IPS_panel" and "Screen-Full_HD".
dummy_variable_1 = pd.get_dummies(df_lt['Screen'])
dummy_variable_1.head()
df_lt = pd.concat([df_lt,dummy_variable_1], axis = 1)
print(df_lt.head())
```
![image](https://github.com/user-attachments/assets/de98bb37-b2cc-4d95-b26a-49d6e8fee8e4)

```
# save data after clean
df_lt.tosave(D:/Data anlysis- working sheet/python/data/laptop_price.csv)
```
# DATA ANALYSE

```
# Install Required Libraries
import piplite
await piplite.install('seaborn')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
%matplotlib inline

# import csv file
df = pd.read_csv('D:/Data anlysis- working sheet/python/data/laptop_price.csv')
```
###  Visualize individual feature patterns

```
# calculate the correlation between variables of type "int64" or "float64"
numeric_df = df.select_dtypes(include=['int64', 'float64'])
numeric_df.corr(
```
![image](https://github.com/user-attachments/assets/01be9e9b-47ec-483b-aac3-d513a735875c)

```
# creates a heatmap to visualize the correlation matrix 
mask = (corr_matrix < 0.7) & (corr_matrix > -0.7)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, mask=mask)

plt.title("Highlighted Correlation Heatmap", fontsize=15)
plt.show()
```
![image](https://github.com/user-attachments/assets/c0f2c533-8ee9-4dd8-a035-ae83e0720016)

In the heatmap, only RAM_GB (~0.55) has a clear correlation with Price. Other factors may not be prominent or not fully displayed in the image.

```
# performs a regression plot analysis for multiple features against "Price" and calculates the correlation between them
column_list = ['CPU_frequency', 'CPU_core', 'Storage_GB_SSD', 'RAM_GB']
for column in column_list:
    sns.regplot(x=column, y="Price", data=df)
    plt.ylim(0,)
    plt.show()
    print(df[[column, 'Price']].corr())
```
![image](https://github.com/user-attachments/assets/bd2fd4f8-fe45-4d5f-bfdd-635caec218b1)
![image](https://github.com/user-attachments/assets/4d3b0543-4552-48cb-af91-cb0ed568d442)
![image](https://github.com/user-attachments/assets/29a0f6f2-8c00-423d-ba01-3f930a6e84c8)
![image](https://github.com/user-attachments/assets/e8b90ea3-807c-4668-922b-282bc936d549)

CPU Frequency vs. Price (Correlation: 0.3666)
A weak to moderate positive correlation.
Higher CPU frequency slightly increases price but is not a dominant factor.
Price variation suggests other factors also contribute significantly.

CPU Cores vs. Price (Correlation: 0.4594)
Moderate positive correlation.
More cores tend to increase price, but clustering at specific core counts (3, 5, 7) suggests limited options.

Storage (SSD) vs. Price (Correlation: 0.2434)
Weak correlation.
Higher SSD capacity slightly increases price, but the effect is minimal.
Most laptops fall into two storage categories (likely 128GB and 256GB).

RAM vs. Price (Correlation: 0.5493)
Strongest correlation among the four factors.
More RAM significantly increases price.
High variability suggests premium models may have additional features beyond RAM.

Key Takeaways
RAM is the most influential factor on price in this dataset.
CPU cores also play a significant role, though the impact is slightly lower.
CPU frequency has some effect but is not the strongest predictor.
Storage has the weakest impact, meaning price is more influenced by other specifications.

```
# Generate Box plots for the different feature that hold categorical values. These features would be "Category", "GPU", "OS", "CPU_core", "RAM_GB", "Storage_GB_SSD"
feature_list = ["Category", "GPU", "OS", "CPU_core", "RAM_GB", "Storage_GB_SSD"]
for column in feature_list:
    sns.boxplot(x=column, y="Price", data=df)
    plt.show()
```
![image](https://github.com/user-attachments/assets/a0f8f476-945c-43be-8c49-f5b20408b7a6)
![image](https://github.com/user-attachments/assets/df5b5505-c49e-400d-8cc0-11acb1a97178)
![image](https://github.com/user-attachments/assets/fc2af7c8-956f-492e-9a45-3ec5ed2a0993)
![image](https://github.com/user-attachments/assets/02ffc294-8755-4c30-904c-25f6d65d88a4)
![image](https://github.com/user-attachments/assets/dc6b7a3b-b00f-4b39-a905-6985b31fa992)
![image](https://github.com/user-attachments/assets/7dd880a5-40eb-4a97-bf5f-b9c3d00838ee)

Category vs. Price

Different categories exhibit varying price distributions.
Some categories have a wide range with outliers, while others show more consistent pricing.
GPU vs. Price

Laptops with higher-tier GPUs tend to have higher median prices.
Significant price variability exists within each GPU category, suggesting other factors also influence pricing.
OS vs. Price

Some OS categories have consistently lower prices.
A wider spread of prices is observed in certain OS categories, possibly indicating premium models.
CPU Cores vs. Price

More CPU cores generally correlate with higher laptop prices.
Laptops with 7-core CPUs have a wider price range, suggesting premium and budget-friendly options exist within this category.
RAM (GB) vs. Price

Higher RAM configurations tend to increase laptop prices.
16GB RAM models have the highest median and price spread.
Storage (SSD GB) vs. Price

Laptops with 256GB SSDs are generally priced higher than those with 128GB SSDs.
Price variation within each category indicates additional contributing factors beyond storage.
Summary
Key hardware specifications (GPU, CPU, RAM, Storage) strongly influence price trends.
Some categories show price outliers, possibly indicating premium models.
More advanced configurations generally result in higher prices, with significant variation within each category.

## Descriptive Statistical Analysis


```
# Generate the statistical description of all the features being used in the data set. Include "object" data types as well
print(df.describe())
print(df.describe(include=['object']))
```
![image](https://github.com/user-attachments/assets/2f6af285-7a1a-4bdd-b2c9-a2732e40ebb1)

```
# GroupBy the price by 'GPU','CPU_core'
df_gptest = df[['GPU','CPU_core','Price']]
grouped_test1 = df_gptest.groupby(['GPU','CPU_core'],as_index=False).mean()
print(grouped_test1)
```
![image](https://github.com/user-attachments/assets/071dbd9f-a402-45dd-bd6e-9480b8a03a1b)

```
# pivot table 'price' between 'GPU','CPU_core'
grouped_pivot = grouped_test1.pivot(index='GPU',columns='CPU_core')
print(grouped_pivot)
```
![image](https://github.com/user-attachments/assets/5d8c5111-06bd-44d4-b53f-0c7ff41f7cd0)

```
# creates a heatmap using pcolor to visualize a pivot table (grouped_pivot). 
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, labels=grouped_pivot.columns, rotation=45, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, labels=grouped_pivot.index, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

fig.colorbar(im)
plt.show()
```
![image](https://github.com/user-attachments/assets/0f54ab47-5d8f-47ba-9f2e-318688f92cbd)
Highest value (~1900, dark blue) appears when a specific GPU is paired with a 7-core CPU.
Lowest value (~800, dark red) occurs when the GPU is paired with a 3-core CPU.
Trend: As the number of CPU cores increases, GPU performance also tends to improve.
Key Insights & Suggestions:

Dependency Between GPU & CPU Cores: GPUs seem to perform better with more CPU cores.

```
# Use the scipy.stats.pearsonr() function to evaluate the Pearson Coefficient and the p-values for each parameter tested above. This will help to determine the parameters most likely to have a strong effect on the price of the laptops.
for param in ['RAM_GB','CPU_frequency','Storage_GB_SSD','Weight_pound','CPU_core','OS','GPU','Category']:
    pearson_coef, p_value = stats.pearsonr(df[param], df['Price'])
    print(param)
    print("The Pearson Correlation Coefficient for ",param," is", pearson_coef, " with a P-value of P =", p_value)
```
![image](https://github.com/user-attachments/assets/62762ae1-d6d2-4246-a334-c81b19ba02ce)

















# Prediction
predicts laptop prices based on four independent variables:

CPU_frequency
CPU_core
Storage_GB_SSD
RAM_GB
```
import statsmodels.api as sm
```

```
X = df[['CPU_frequency', 'CPU_core', 'Storage_GB_SSD', 'RAM_GB']]  # Independent variables
y = df['Price']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()  # Fit the model
print(model.summary())  # Display regression summary
```
![image](https://github.com/user-attachments/assets/023d5b37-c80f-414f-b46e-1ef3992bc18e)
Metric	Value	Meaning
R-squared	0.400	The model explains 40% of the variation in price. This is a moderate but not strong predictive power.
Adj. R-squared	0.389	Adjusted R² is slightly lower, accounting for the number of predictors. This suggests that adding more features might improve the model.
F-statistic (38.76, p = 7.45e-25)	The low p-value (< 0.05) means that at least one predictor is statistically significant. The model is better than random guessing.	
Log-Likelihood (-1788.7)	Lower values mean a worse fit. Used for comparing models.	
AIC (3587) & BIC (3605)	Lower values are better when comparing models.	

Coefficients (Impact of Each Feature on Price)
Each coefficient represents how much the price changes for a one-unit increase in that feature, while keeping other factors constant.

| Feature | Coef | Std Err | t | p-value (P>|t|) | Interpretation | |---------|--------|--------|------|--------------|----------------| | Intercept (const) | -586.69 | 264.26 | -2.22 | 0.027 | The model predicts a baseline price of $-586.69, but this has no real meaning. | | CPU_frequency | 910.35 | 214.89 | 4.24 | 0.000 | Significant: A 1 GHz increase in CPU frequency increases price by $910. | | CPU_core | 99.88 | 28.22 | 3.53 | 0.000 | Significant: Adding one more CPU core increases price by $99.88. | | Storage_GB_SSD | 0.1114 | 0.942 | 0.118 | 0.906 | Not significant: SSD storage does not significantly affect price. | | RAM_GB | 91.14 | 138.06 | 0.66 | 0.602 | Not significant: RAM does not have a significant effect on price. |

Key Insights
✅ CPU_frequency and CPU_core are significant predictors of price.
❌ RAM_GB and Storage_GB_SSD do not significantly affect price in this model.
📉 Storage and RAM may be correlated with other features (multicollinearity issue).

 Scatter Plot of Actual vs. Predicted Prices
 If the points align along the y = x line, the model is accurate.
Large deviations indicate the model over-predicts or under-predicts certain price points.
```
sns.scatterplot(x=df['Price'], y=df['Predicted_Price'])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Laptop Prices")
plt.show()
```
![image](https://github.com/user-attachments/assets/9272b9ac-0509-4747-8521-cdae325dbbb7)

Residual Plot (Errors in Predictions)
```
residuals = df['Price'] - df['Predicted_Price']
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals (Actual - Predicted)")
plt.title("Residual Distribution")
plt.show()
```
![image](https://github.com/user-attachments/assets/bd24dd8c-c149-4f47-ad90-8673dfab7426)
 OLS Regression Results (First Image)
This is the summary output of an Ordinary Least Squares (OLS) regression model predicting laptop prices using several features.

Key Metrics:
R-squared = 0.400 → The model explains 40% of the variance in laptop prices, meaning it has moderate predictive power.
Adj. R-squared = 0.389 → Adjusted for the number of predictors, slightly lower but still indicates a moderate fit.
F-statistic = 38.76, p-value = 7.45e-25 → The model is statistically significant, meaning at least one predictor has an effect on price.
Cond. No. = 2.6e+03 → A high condition number suggests potential multicollinearity issues (correlated independent variables).
Interpreting the Coefficients:
Variable	Coefficient (coef)	p-value	Interpretation
Intercept (const)	-586.69	0.027	The base price when all features are zero (not meaningful in this case).
CPU_frequency	910.35	0.000	Increasing CPU frequency by 1 unit increases price by 910.35, statistically significant.
CPU_core	99.88	0.000	More CPU cores increase price, statistically significant.
Storage_GB_SSD	0.111	0.906	No significant effect on price (p > 0.05).
RAM_GB	91.14	0.002	More RAM increases price, statistically significant.
Key Findings:

CPU frequency, CPU cores, and RAM significantly impact price.
SSD storage is not a significant predictor in this model.
Multicollinearity is likely an issue (condition number is large).
2. Scatter Plot - Actual vs. Predicted Prices (Second Image)
Ideally, points should lie on the diagonal line (Actual Price = Predicted Price).
However, the spread suggests that the model doesn't predict well for high-end laptops (over 2500).
Potential issue: The model underestimates expensive laptops.
3. Residual Distribution (Third Image)
Residuals = Actual Price - Predicted Price.
The distribution is roughly normal but slightly skewed.
Issue: Some large residuals indicate model errors, suggesting non-linearity or missing variables.
Conclusions & Next Steps
Improve Model Performance:

Include interaction terms or polynomial features.
Try non-linear models (e.g., Decision Trees, Random Forest).
Reduce multicollinearity using Principal Component Analysis (PCA) or removing correlated variables.
Evaluate Model on New Data:

Use cross-validation to ensure generalizability.
Consider feature engineering to better capture laptop pricing trends.


