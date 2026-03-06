# NUMPY OPERATIONS
 

# Load the modules
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

# ARRAY MANIPULATION
# Creating a 2 D Array
arr = np.array(np.random.randint(20,90,(10,10)))
print(f"The defined array is \n{arr}")
print(f" \n The shape of the array is {arr.shape}")

# Storing the transpose in another matrix object
andes = np.transpose(arr)
print(f" The transpose of the array is \n{andes}")
print(f" \n The shape of the array is {andes.shape}")

# 3D Array
random_3d_array = np.array(np.random.randint(20,90,(3,2,4)))
print(f"The defined array is \n{random_3d_array}")
print(f" \n The shape of the array is {random_3d_array.shape}")

# Storing the transpose of the created 3d array
transposed_3d_array = np.transpose(random_3d_array)
print(f" The transpose of the array is \n{transposed_3d_array}")
print(f" \n The shape of the array is {transposed_3d_array.shape}")

# RESHAPING OF AN ARRAY
# Create a two-dimensional array
cohort = np.array(np.random.randint(20,90,(2,8))) # 2D
print(f"The defined array is \n{cohort}")
print(f" \n The shape of the array is {cohort.shape}")

# reshape this to 1-D,2-D or 3-D array
new_shape = (2,4,2)
reshaped_cohort = np.reshape(cohort,new_shape)
print(f"The reshaped array is \n{reshaped_cohort}")
print(f" \n The shape of the array is {reshaped_cohort.shape}")

# RESIZING AN ARRAY
# Intialize a 3D array
r3D_cohort = np.array(np.random.randint(20,90,(2,4,6)))
print(f"The defined array is \n{r3D_cohort}")
print(f" \n The shape of the array is {r3D_cohort.shape}")

# Resize the created 3D array
new_shape = (2,3,6)
resized_r3D_cohort = np.resize(r3D_cohort,new_shape)
print(f"The resized array is \n{resized_r3D_cohort}")
print(f" \n The shape of the array is {resized_r3D_cohort.shape}")

r3D_cohort.resize(new_shape)
r3D_cohort

# FLATTENED AN ARRAY
# Intialize a 3D array
r3D_cohort = np.array(np.random.randint(20,95,(2,4,6)))
print(f"The defined array is \n{r3D_cohort}")
print(f" \n The shape of the array is {r3D_cohort.shape}")

# Flatten the 3D array
r3D_cohort.flatten()

#Flattens the array in row-major order with 'C'
r3D_cohort.flatten(order='F')

# INSERTING ALONG AXIS
a = np.random.randint(20,30,(2,3,4))    # Axis 0 - 2 ; Axis 1 - 3 ; Axis 2 - 4
print(a)
print(a.shape)

#Insert along axis =0
np.insert(a,1,100,axis=0)

# Insert along axis = 1
np.insert(a,0,50,axis=1)

# Insert along axis=2
np.insert(a,0,200,axis=2)

a = np.random.randint(2,8,(2,4,4))
print(a)
print(a.shape)

# APPEND AN ARRAY
# Append a value to the array. It flattens the array and append -99 to the end
np.append(a, 50)

# Fix the axis other than axis of appending
b = np.random.randint(2,8,(2,1,4))
print(b)
print(a)
print(a.shape)

np.append(a,b,axis = 1)

a = np.random.randint(2,4,(2,3,2))
print(a)
print(a.shape)

# DELETE
# Delete along Axis=0
np.delete(a,(0),axis = 0)

# Delete along axis=1
np.delete(a,(0,2),axis = 1)
a

# Delete along axis=2
np.delete(a,(0,1),axis = 2)

# UNIQUE
#Initialized array
a = np.random.randint(2,20,(2,4,3))
print(a)
print(a.shape)

np.unique(a)

# return_counts= True
np.unique(a,return_counts=True) # Returns tuple

# return_index = True
np.unique(a,return_counts=True,return_index=True)

# COPY AND VIEW
# Lets take an example of a list
a = [1,2,3,4,5,6]
b = a
b[0] = 10
print(b)
print(a)

# Lets use the copy method
a = [1,2,3,4,5,6]
b = a.copy()
b[0] = 10
print(b)
print(a)

a = np.array([2,34,12])
b = a
b[0] = -999
print(b)
print(a)

a = np.array([2,34,12])
b = a.copy()
b[0] = -999
print(b)
print(a)

# BROADCASTING
np.arange(3) + 5
np.ones((3,3)) + np.arange(3)
np.arange(3).reshape((3,1)) + np.arange(3)
np.arange(3).reshape((3,1)) + np.arange(4)
np.arange(2)

#Example 2
a=np.random.randint(2,6,(2,4))
print(a)
b=np.random.randint(1,10,(2,2))
print(b)

# MULTIPLY TWO ARRAYS
a=np.random.randint(2,6,(2,4))
print(a)
b=np.random.randint(0,5,(2,4))
print(b)

np.multiply(a,b)

# DIVIDE TWO ARRAY
np.divide(a,b)

a=np.random.randint(5,9,(2,4))
print(a)

print(np.sum(a))
print(np.sum(a,axis=0))
print(np.sum(a,axis=1))

# Calculate minimum
print(np.min(a))
print(np.min(a,axis=1)) # Across Column

# Sort the array
a=np.random.randint(0,20,(3,3))
print(a)

print(np.sort(a,axis=0)) # Ascending
print(np.sort(a,axis=1))
print(-np.sort(-a,axis=1)) # Descending

sum(np.arange(3)*np.arange(3).transpose())


# ==========================================
# PANDAS OPERATIONS
# ==========================================



# READ DATA
# loading csv File
df = pd.read_csv('/content/International_T20_Data.csv', engine='python', on_bad_lines='skip')

df.shape
df.head()
df.tail(10)

# DESCRIBE
df.describe()

# INDEXING AND SLICING
df.head()

# iloc FUNCTION GETS ROWS AND COLUMN AT PARTICULAR FUNCTION
#subset which consists of first 8 rows and first 4 columns
df.iloc[:,-3:]

# loc DUNCTION GETS ROWS AND COLUMNS WITH PARTICULAR LABELS
# subset dataframe in which we want to have first 8 rows identified with their rows labels and some named columns
df.loc[50:60,['innings', 'meta.data_version', 'meta.created']]

df.columns
df.loc[:,['innings', 'meta.data_version', 'meta.created']]
df[0:20]
df[['info.dates', 'info.gender', 'info.match_type',]]
df.columns

# BOOLEAN INDEXING
df['info.dates'] == 'en'

# Boolean Indexing
gender = df[df['info.gender'] == 'en']
gender.head()
gender.shape

day = df[df['info.dates'] != 'en']
day.head()

# MANIPULATE COLUMNS
##COLUMN OPERATIONS
df.rename(columns={'info.city':'City'}, inplace=True)
df['Runs_margin'] = df['info.outcome.by.runs']
df.sort_values(by='Runs_margin', ascending=False).head()
df.shape
df['Runs_margin']
df.head()

# DROP COLUMN
df.drop(['meta.data_version'], axis = 1, inplace=True)
df.head()
df.describe()
df.head()

# BASIC OPERATIONS
# if null value
df.loc[:1,'innings'] = np.nan
df.iloc[:5]
df.head()
df.iloc[:3]
df.head()

# APPLY FUNCTION
# Suppose we want to convert the column network to uppercase
def to_uppercase(column):
  return column.upper()

df.head()

# APPLYING apply() FUNCTION USING LAMBDA FUNCTION
def add_two_numbers(a,b):
  return a+b

x = lambda a,b : a+b
x(1,5)

df.head()

# SORT SOME DATA COLUMNS IN ASCENDING OR DESCENDING ORDER
# Sort descending order
df.sort_values('Runs_margin', ascending=False)

# DATE AND TIME OPERATIONS &B FUNCTION
df.info()
df['Runs_margin'][0:5]

# Import datetime modules
from datetime import datetime
from datetime import date

# strptime() AND Strftime() FUNCTION
df['Runs_margin'][0:5]
df.info()
df.head()
df['Runs_margin'][0]

print(date.today())
print(datetime.now())

from datetime import timedelta
datetime.now() + timedelta(days=1)

# operations on datetime
from datetime import timedelta
print(date.today())
print(date.today() - timedelta(days=1))
print(datetime.now() + timedelta(hours=5.5))
print(datetime.now() + timedelta(seconds=60))

print(df['Runs_margin'][0])
print(df['Runs_margin'][5])

t1 = pd.to_datetime('1/1/2015 01:00')
t2 = pd.to_datetime('1/1/2015 03:30')
print(pd.Timedelta(t2 - t1).seconds / 3600.0)

# BASIC OPERATIONS
df.info()

# list of unique values in a column
list(df['Runs_margin'].unique())

# Counts  number of unique values in column
df.Runs_margin.nunique()

# Give  counts of  category of variable
df['Runs_margin'].value_counts()

# Returns series of booleans if a column value = null
df['Runs_margin'].isnull()
df[~df['Runs_margin'].isnull()]

# fill the missing values with value -999
df['Runs_margin'].fillna('Not Present',inplace = True)
df.head()


# ==========================================
# METPLOTLIB / SEABORN OPERATIONS
# ==========================================

# Import pandas


df = pd.read_csv('/content/International_T20_Data.csv', engine='python', on_bad_lines='skip')
df.head()
df.shape

# %lsmagic (Jupyter magic commands)

from datetime import datetime
# Importing matplotlib.pyplot
import matplotlib.pyplot as plt
%matplotlib inline

df.head()

df.rename(columns={'info.city':'City'}, inplace=True)
df['Runs_margin'] = df['info.outcome.by.runs']
mon = df["Runs_margin"].value_counts()
type(mon)
mon.sort_index(inplace=True)

plt.rcParams['figure.figsize'] = (10, 5)

##DATA CLEANING
##-MISSING VALUES
df.isnull().sum()
# df.dropna(inplace=True) # Commenting out to prevent loss of all data

##Removing duplicates
df.duplicated().sum()
df.drop_duplicates(inplace=True)

##COLUMN OPERATIONS
df.rename(columns={'info.city':'City'}, inplace=True)
df['Runs_margin'] = df['info.outcome.by.runs']
df.sort_values(by='Runs_margin', ascending=False).head()

# Fix: Convert 'info.outcome.by.runs' to numeric, coercing errors
df['info.outcome.by.runs'] = pd.to_numeric(df['info.outcome.by.runs'], errors='coerce')

##Filtering Data
df[df['info.gender'] == 'male']
df[df['info.outcome.by.runs'] > 50]
df.loc[:, ['City','info.venue']]

#GroupBy Operations
##Matches per city
df.groupby('City').size()

##Average run margin by venue
df.groupby('info.venue')['info.outcome.by.runs'].mean()

##Matches won by each team
df['info.outcome.winner'].value_counts()

runs = np.array(df['info.outcome.by.runs'])

np.mean(runs)
np.median(runs)
np.std(runs)
np.var(runs)
np.max(runs)
np.min(runs)
np.random.choice(runs,10)

#Pivot Table
pd.pivot_table(df,
               values='info.outcome.by.runs',
               index='City',
               aggfunc=np.mean)

# MATPLOTLIB VISUALIZATIONS

# BAR CHART
teams = df['info.outcome.winner'].value_counts().head(10)
plt.figure(figsize=(10,6))
plt.bar(teams.index, teams.values)
plt.xticks(rotation=45)
plt.title("Top Teams by Wins")
plt.show()

# HISTOGRAM
plt.hist(df['info.outcome.by.runs'], bins=20)
plt.title("Run Margin Distribution")
plt.xlabel("Runs")
plt.ylabel("Frequency")
plt.show()

# PIE CHART
gender = df[df['info.gender'].isin(['male', 'female'])]['info.gender'].value_counts()
plt.pie(gender.values, labels=gender.index, autopct='%1.1f%%')
plt.title("Match Distribution by Gender")
plt.show()


# SEABORN VISUALIZATION

# COUNT PLOT
plt.figure(figsize=(12, 6))
toss_decisions = df['info.toss.decision'].value_counts()
plt.bar(toss_decisions.index, toss_decisions.values, width=0.6) # Adjust 'width' as desired
plt.title("Toss Decisions")
plt.xlabel("Decision")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# BOX PLOT
plt.figure(figsize=(10, 6))
sns.boxplot(x=df[df['info.gender'].isin(['male', 'female'])]['info.gender'], y=df['info.outcome.by.runs'], width=0.5)
plt.title("Distribution of Run Margins by Gender")
plt.xlabel("Gender")
plt.ylabel("Run Margin")
plt.show()

# LINE PLOT
plt.figure(figsize=(8,5))

# Ensure 'info.overs' is numeric and handle NaNs for a clean line plot
df['info.overs_numeric'] = pd.to_numeric(df['info.overs'], errors='coerce')
plot_data = df.dropna(subset=['info.overs_numeric'])

# Use lineplot with errorbar=None to remove the blue fill
sns.lineplot(x=plot_data.index, y=plot_data['info.overs_numeric'], errorbar=None)

plt.title("Line Plot of Overs")
plt.xlabel("Index")
plt.ylabel("Overs") # Set y-axis label explicitly
plt.show()

# MULTIPLE LINE PLOT IN SAME FIGURE
plt.figure(figsize=(8,5))

df['info.overs_numeric'] = pd.to_numeric(df['info.overs'], errors='coerce')
df['meta.revision_numeric'] = pd.to_numeric(df['meta.revision'], errors='coerce')

plt.plot(df['info.overs_numeric'].dropna(), label="Overs")
plt.plot(df['meta.revision_numeric'].dropna(), label="Revision")

plt.legend()
plt.title("Multiple Line Plot")

plt.show()

# VIOLIN PLOT
sns.violinplot(x=df[df['info.gender'].isin(['male', 'female'])]['info.gender'], y='info.outcome.by.runs', data=df, width=0.8)
plt.xlabel("Gender")
plt.title("Distribution of Run Margins by Gender (Violin Plot)")
plt.show()

# DISPLOT
df['info.overs_numeric'] = pd.to_numeric(df['info.overs'], errors='coerce')
sns.displot(df['info.overs_numeric'].dropna(), kde=True, bins=10, height=6, aspect=1.5) # Adjust bins, height, and aspect for visual effect
plt.title("Distribution Plot of Overs")
plt.xlabel("Overs") # Set x-axis label explicitly
plt.show()

# UNDERSTANDING CORRELATION
##Correlation Analysis
df.corr(numeric_only=True)

numeric_df = df.select_dtypes(include=np.number)
# Correlation matrix
corr = numeric_df.corr()
print(corr)

# CORRELATION HEATMAP
plt.title(" CORREALTION HEATMAP ")
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.figure(figsize=(2,2))