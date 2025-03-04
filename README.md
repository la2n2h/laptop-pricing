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

# Loading data
### Load the dataset to a pandas dataframe named 'df'
```
import pandas as pd
import numpy as np
```
```
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"
```
```
df = pd.read_csv(file_path, header=None)
df.head()
```
![image](https://github.com/user-attachments/assets/aff94d6a-d25f-436c-8fff-e47064a5094c)

change the header names
```
df.columns = ["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core", "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg", "Price"]
df.head()
```
![image](https://github.com/user-attachments/assets/0d8a996d-3f57-4b59-917d-ee4deeed9a24)

Replace '?' with 'NaN'
```
df.replace('?' , np.NaN)
```

print the data types of df columns
```
df.dtypes
```
![image](https://github.com/user-attachments/assets/edd6878f-8829-49de-95cd-4d693b593ca8)

Print the statistical description of the dataset, including that of 'object' data types.¶
```
df.describe(include ='all')
```
![image](https://github.com/user-attachments/assets/b2694f48-2772-4e7b-a7aa-db91f88a306e)

Print the summary information of the dataset.
```
df.info()
```
![image](https://github.com/user-attachments/assets/53f9640e-e348-4ac0-b780-d23ea6e523ed)


# Pre-processing 


