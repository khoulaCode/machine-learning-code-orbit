# Pandas: A Comprehensive Guide

This document provides a detailed overview of the Pandas library, an essential tool for data manipulation and analysis in Python.

---

## 1. Introduction to Pandas

Pandas is an open-source Python library built on top of NumPy, designed for data manipulation and analysis. It provides high-performance, easy-to-use data structures and data analysis tools. Pandas is particularly well-suited for working with tabular data (like spreadsheets or SQL tables) and time series data.

### Why Pandas?

- **Efficient Data Handling**: Provides fast and flexible data structures.
- **Missing Data Handling**: Built-in functionalities to deal with missing data.
- **Data Alignment**: Automatically aligns data based on labels, preventing common errors.
- **Flexible GroupBy Functionality**: Powerful tools for splitting, applying, and combining datasets.
- **Merging and Joining**: Intuitive methods for combining datasets.
- **Time Series Functionality**: Robust tools for working with date and time-indexed data.
- **Integration**: Seamlessly integrates with other Python libraries like NumPy, Matplotlib, and Scikit-learn.

---

## 2. NumPy Array vs. Pandas DataFrame

| Feature            | NumPy Array                                          | Pandas DataFrame                                                                 |
|--------------------|------------------------------------------------------|----------------------------------------------------------------------------------|
| **Data Structure** | Homogeneous (same data type)                         | Heterogeneous (columns can have different data types)                           |
| **Indexing**       | Integer-based (0, 1, 2...)                           | Labeled (row labels, column names)                                              |
| **Dimensions**     | 1D, 2D, 3D, or N-dimensional                         | Primarily 2D                                                                    |
| **Column Names**   | No inherent concept                                  | Explicit column names                                                           |
| **Missing Data**   | Manual filtering required                            | Built-in methods like `dropna()`, `fillna()`                                    |
| **Use Case**       | Numerical/scientific computing                       | Tabular data manipulation and analysis                                          |
| **Flexibility**    | Less flexible for mixed-type data                    | Highly flexible for structured data                                             |

**Analogy**:
- NumPy array: like a mathematical matrix.
- Pandas DataFrame: like a spreadsheet or SQL table with labeled axes.



## 3. Series and DataFrame Creation

### 3.1 Pandas Series

A Pandas Series is a one-dimensional labeled array.

#### From a Python List:
```python
import pandas as pd
my_list = [10, 20, 30, 40]
s = pd.Series(my_list)
s
```
With Custom Index:
```python
labels = ['a', 'b', 'c', 'd']
s_indexed = pd.Series(data=my_list, index=labels)
print(s_indexed)
```
From a NumPy Array:
python
```python
import numpy as np
arr = np.array([100, 200, 300])
s_np = pd.Series(arr)
print(s_np)
```
From a Dictionary:
```python

d = {'a': 10, 'b': 20, 'c': 30}
s_dict = pd.Series(d)
print(s_dict)
```
### 3.2 Pandas DataFrame
A DataFrame is a 2D labeled data structure.

From a Dictionary:
```python

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'London', 'Paris', 'Tokyo']
}
df = pd.DataFrame(data)
```
From a List of Dictionaries:
```python

data_list_of_dicts = [
    {'Name': 'Eve', 'Age': 28, 'City': 'Berlin'},
    {'Name': 'Frank', 'Age': 32, 'City': 'Rome'}
]
df_lod = pd.DataFrame(data_list_of_dicts)
```
From a NumPy 2D Array:
```python
np_array = np.array([[1, 2, 3], [4, 5, 6]])
df_np = pd.DataFrame(np_array, columns=['ColA', 'ColB', 'ColC'], index=['Row1', 'Row2'])
```

## 4. Indexing and Slicing
Indexing and slicing are fundamental for selecting specific rows, columns, or subsets of data from Series and DataFrames.
### 4.1 Series Indexing and Slicing
By Label:

```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s['b'])
print(s['a':'c'])  # Inclusive of 'c'
```
By Position (Integer Index):
```python

print(s[0])
print(s[0:3])  # Exclusive of index 3
```

### 4.2 DataFrame
DataFrames have both row and column labels. Pandas provides two primary methods for indexing: loc (label-based) and iloc (integer-location based).

Let's use the `df` DataFrame created earlier:
```py
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 45],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin'],
    'Salary': [50000, 60000, 75000, 80000, 90000]
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)
```
Selecting Columns:
```py
# Select a single column (returns a Series)
print("\n'Name' column:\n", df['Name'])

# Select multiple columns (returns a DataFrame)
print("\n'Name' and 'City' columns:\n", df[['Name', 'City']])
```
`loc` (Label-based Indexing):
Used for selecting data by explicit label of rows and columns.
Syntax: `df.loc[row_label, column_label]`
```py
# Select row with index 0
print("\nRow at index 0 (using loc):\n", df.loc[0])

# Select rows with index 0 and 2
print("\nRows at index 0 and 2 (using loc):\n", df.loc[[0, 2]])

# Select rows 0 to 2 (inclusive) and 'Name' column
print("\nRows 0-2, 'Name' column (using loc):\n", df.loc[0:2, 'Name'])

# Select rows 0 to 2 (inclusive) and 'Name' to 'City' columns (inclusive)
print("\nRows 0-2, 'Name' to 'City' columns (using loc):\n", df.loc[0:2, ['Name', 'City']])
```
`iloc` (Integer-location based Indexing):
Used for selecting data by integer position of rows and columns (0-based).
Syntax: `df.iloc[row_position, column_position]`

```py
# Select row at position 0
print("\nRow at position 0 (using iloc):\n", df.iloc[0])

# Select rows at positions 0 and 2
print("\nRows at positions 0 and 2 (using iloc):\n", df.iloc[[0, 2]])

# Select rows 0 to 2 (exclusive) and column at position 0
print("\nRows 0-2 (exclusive), column at pos 0 (using iloc):\n", df.iloc[0:3, 0])

# Select rows 0 to 2 (exclusive) and columns at positions 0 to 2 (exclusive)
print("\nRows 0-2 (exclusive), columns 0-2 (exclusive) (using iloc):\n", df.iloc[0:3, 0:3])
```
Boolean Indexing/Selection:
Used to filter data based on a condition. Returns rows where the condition is `True`.
```py
# Select rows where Age is greater than 30
print("\nPeople older than 30:\n", df[df['Age'] > 30])

# Select Name and City for people older than 30
print("\nName and City for people older than 30:\n", df[df['Age'] > 30][['Name', 'City']])

# Multiple conditions (use & for AND, | for OR)
print("\nPeople older than 30 AND living in London/Paris:\n", df[(df['Age'] > 30) & ((df['City'] == 'London') | (df['City'] == 'Paris'))])
```
## 5. Basic Operations
### 5.1 Arithmetic Operations
Pandas allows element-wise arithmetic operations on Series and DataFrames. Operations are aligned by index.
```py
df['Age_Plus_5'] = df['Age'] + 5
print("\nDataFrame with 'Age_Plus_5' column:\n", df)

df['Salary_in_K'] = df['Salary'] / 1000
print("\nDataFrame with 'Salary_in_K' column:\n", df)
```
### 5.2 Unique Values and Value Counts
`unique()`: Returns unique values in a Series.

`nunique()`: Returns the number of unique values.

`value_counts()`: Returns a Series containing counts of unique values.
```py
df['City'].unique() # Output: array(['New York', 'London', 'Paris', 'Tokyo', 'Berlin'], dtype=object)
df['City'].nunique() # Output: 5
df['City'].value_counts()
# Output:
# City
# New York    1
# London      1
# Paris       1
# Tokyo       1
# Berlin      1
# Name: count, dtype: int64
```
### 5.3 Handling Missing Data
Missing data is often represented as `NaN` (Not a Number).

- `isnull()`: Returns a boolean DataFrame/Series indicating where values are NaN.

- `notnull()`: Returns a boolean DataFrame/Series indicating where values are NOT NaN.

- `dropna()`: Drops rows/columns containing missing values.

    - `axis=0` (default): drops rows.

    - `axis=1`: drops columns.

    - `how='any'` (default): drops if any NaN is present.

    - `how='all'`: drops if all values are NaN.

- `fillna()`: Fills missing values with a specified value.
```py
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})
print("\nDataFrame with missing values:\n", df_missing)

print("\nIs null:\n", df_missing.isnull())

# Drop rows with any NaN
print("\nAfter dropping rows with NaN:\n", df_missing.dropna())

# Drop columns with any NaN
print("\nAfter dropping columns with NaN:\n", df_missing.dropna(axis=1))

# Fill NaN with a specific value (e.g., 0)
print("\nAfter filling NaN with 0:\n", df_missing.fillna(0))

# Fill NaN in 'A' with mean
print("\nAfter filling NaN in 'A' with mean:\n", df_missing['A'].fillna(df_missing['A'].mean()))
```
## 6. Data Input and Output (I/O)
Pandas provides functions to read and write data from various file formats.

- **CSV (Comma Separated Values):**

    - `pd.read_csv('file.csv')`: Reads data from a CSV file into a DataFrame.

    - `df.to_csv('output.csv', index=False)`: Writes a DataFrame to a CSV file. index=False prevents writing the DataFrame index as a column.

- **Excel:**

    - `pd.read_excel('file.xlsx', sheet_name='Sheet1')`: Reads data from an Excel file.

    - `df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)`: Writes a DataFrame to an Excel file.

- **SQL Databases:**

    - `pd.read_sql_table('table_name', con=engine)`: Reads a SQL table.

    - `pd.read_sql_query('SELECT * FROM table_name', con=engine)`: Reads data using a SQL query.

    - `df.to_sql('table_name', con=engine, if_exists='append', index=False)`: Writes a DataFrame to a SQL table. (Requires SQLAlchemy and database-specific drivers).

- **JSON (JavaScript Object Notation):**

    - `pd.read_json('file.json')`

    - `df.to_json('output.json')`

- **HTML Tables:**

    - `pd.read_html('url_or_file.html')`: Reads HTML tables into a list of DataFrames.

## 7. Data Inspection and Exploration
Once data is loaded, it's crucial to inspect and explore it to understand its structure, content, and quality.

- `df.head(n=5)`: Returns the first n rows of the DataFrame (default 5).

- `df.tail(n=5)`: Returns the last n rows of the DataFrame (default 5).

- `df.info()`: Prints a concise summary of the DataFrame, including the data types of columns, non-null values, and memory usage.

- `df.describe()`: Generates descriptive statistics of numerical columns (count, mean, std, min, max, quartiles).

- `df.shape`: Returns a tuple representing the dimensions of the DataFrame (rows, columns).

- `df.columns`: Returns a list of column names.

- `df.index`: Returns the index (row labels) of the DataFrame.

- `df.dtypes`: Returns a Series with the data type of each column.

- `df.value_counts()`: (for Series) Returns counts of unique values.

- `df.corr()`: Computes pairwise correlation of columns, excluding NA/null values.
```py
print("\n--- Data Inspection and Exploration ---")
print("\nFirst 3 rows:\n", df.head(3))
print("\nLast 2 rows:\n", df.tail(2))
print("\nDataFrame Info:")
df.info()
print("\nDescriptive Statistics:\n", df.describe())
print("\nShape of DataFrame:", df.shape)
print("\nColumn names:", df.columns)
print("\nIndex of DataFrame:", df.index)
print("\nData types of columns:\n", df.dtypes)
print("\nValue counts for 'City':\n", df['City'].value_counts())
print("\nCorrelation matrix:\n", df.corr(numeric_only=True))
```
## 8. Data Cleaning and Preprocessing
Data cleaning involves fixing or removing incorrect, corrupted, incorrectly formatted, duplicate, or incomplete data. Preprocessing prepares data for analysis.

- **Handling Missing Data: (Covered in Section 5.3)**

    - `df.dropna()`

    - `df.fillna()`

- **Handling Duplicates:**

    - df.duplicated(): Returns a boolean Series indicating duplicate rows.

    - df.drop_duplicates(): Removes duplicate rows.

        - `subset`: Column(s) to consider for identifying duplicates.

        - `keep='first'` (default): Keep the first occurrence.

        - `keep='last'`: Keep the last occurrence.

        - `keep=False`: Drop all duplicates.
```py
df_dup = pd.DataFrame({
    'A': [1, 2, 2, 3, 4],
    'B': ['x', 'y', 'y', 'z', 'w']
})
print("\nDataFrame with duplicates:\n", df_dup)
print("\nDuplicated rows (boolean Series):\n", df_dup.duplicated())
print("\nAfter dropping duplicates:\n", df_dup.drop_duplicates())
print("\nAfter dropping duplicates based on 'A' keeping first:\n", df_dup.drop_duplicates(subset=['A'], keep='first'))
```
- **Changing Data Types (astype()):**
Useful for converting columns to appropriate data types (e.g., object to numeric, numeric to categorical).
```py
df['Age'] = df['Age'].astype(float) # Convert 'Age' to float
print("\nDataFrame dtypes after converting 'Age' to float:\n", df.dtypes)

# Convert 'Age' back to int (requires no NaN values)
df['Age'] = df['Age'].astype(int)
print("\nDataFrame dtypes after converting 'Age' to int:\n", df.dtypes)
```
- **Renaming Columns:**
```py
df_renamed = df.rename(columns={'Name': 'Full_Name', 'City': 'Location'})
print("\nDataFrame after renaming columns:\n", df_renamed)
```
- **Applying Functions (apply()):**

    `apply()` can be used to apply a function along an axis of the DataFrame or Series.
```py
# Apply a lambda function to a column
df['Age_Category'] = df['Age'].apply(lambda x: 'Adult' if x >= 18 else 'Minor')
print("\nDataFrame with 'Age_Category' column:\n", df)

# Apply a function to multiple columns (axis=1 for row-wise operation)
def calculate_bonus(row):
    if row['Age'] > 30:
        return row['Salary'] * 0.10
    else:
        return row['Salary'] * 0.05

df['Bonus'] = df.apply(calculate_bonus, axis=1)
print("\nDataFrame with 'Bonus' column:\n", df)
```
## 9. Data Manipulation and Transformation
These operations allow you to reshape, combine, and aggregate your data.

### `groupby()`:
Used for grouping rows together based on a column's values and then applying an aggregation function (e.g., `sum()`, `mean()`, `count()`, `min()`, `max()`).
```py
data_sales = {
    'Region': ['East', 'West', 'East', 'West', 'East', 'South'],
    'Product': ['A', 'B', 'A', 'C', 'B', 'A'],
    'Sales': [100, 150, 120, 200, 180, 90]
}
df_sales = pd.DataFrame(data_sales)
print("\nOriginal Sales DataFrame:\n", df_sales)

# Group by 'Region' and calculate total sales
print("\nTotal Sales by Region:\n", df_sales.groupby('Region')['Sales'].sum())

# Group by 'Region' and 'Product' and calculate mean sales
print("\nMean Sales by Region and Product:\n", df_sales.groupby(['Region', 'Product'])['Sales'].mean())

# Using .agg() for multiple aggregations
print("\nMultiple aggregations by Region:\n",
      df_sales.groupby('Region')['Sales'].agg(['sum', 'mean', 'count']))
```
### `merge()`, `join()`, `concat()`:
Used to combine DataFrames.

- `concat()`: Stacks DataFrames either vertically (rows) or horizontally (columns).

    - `axis=0` (default): Stacks rows.

    - `axis=1`: Stacks columns.
```py
df1 = pd.DataFrame({'A': ['A0', 'A1'], 'B': ['B0', 'B1']}, index=[0, 1])
df2 = pd.DataFrame({'A': ['A2', 'A3'], 'B': ['B2', 'B3']}, index=[2, 3])
print("\nDataFrame 1:\n", df1)
print("\nDataFrame 2:\n", df2)

result_concat_rows = pd.concat([df1, df2])
print("\nConcatenated (rows):\n", result_concat_rows)

df3 = pd.DataFrame({'C': ['C0', 'C1'], 'D': ['D0', 'D1']}, index=[0, 1])
result_concat_cols = pd.concat([df1, df3], axis=1)
print("\nConcatenated (columns):\n", result_concat_cols)
```
- `merge()`: Combines DataFrames based on common columns (like SQL JOINs).

    - `on`: Column(s) to merge on.

    - `how`: Type of merge (inner, left, right, outer).

        - `inner` (default): Only rows with matching keys in both DataFrames.

        - `left`: All rows from left DataFrame, plus matching from right.

        - `right`: All rows from right DataFrame, plus matching from left.

        - `outer`: All rows from both DataFrames, filling NaN where no match.
```py
df_customers = pd.DataFrame({
    'CustomerID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David']
})
df_orders = pd.DataFrame({
    'OrderID': [101, 102, 103, 104],
    'CustomerID': [2, 4, 1, 5], # CustomerID 5 doesn't exist in df_customers
    'Amount': [100, 250, 150, 300]
})
print("\nCustomers DataFrame:\n", df_customers)
print("\nOrders DataFrame:\n", df_orders)

# Inner merge
merged_inner = pd.merge(df_customers, df_orders, on='CustomerID', how='inner')
print("\nInner Merge (matching CustomerIDs only):\n", merged_inner)

# Left merge
merged_left = pd.merge(df_customers, df_orders, on='CustomerID', how='left')
print("\nLeft Merge (all customers, matching orders):\n", merged_left)
```
- `join()`: Combines DataFrames based on their indexes (or a column in one and index in another). Often used when DataFrames have a common index.
```py
df_left = pd.DataFrame({'A': ['A0', 'A1'], 'B': ['B0', 'B1']}, index=['K0', 'K1'])
df_right = pd.DataFrame({'C': ['C0', 'C2'], 'D': ['D0', 'D2']}, index=['K0', 'K2'])
print("\nLeft DataFrame for join:\n", df_left)
print("\nRight DataFrame for join:\n", df_right)

joined_df = df_left.join(df_right, how='outer')
print("\nOuter Join (on index):\n", joined_df)
```
- `pivot_table()`:
Creates a spreadsheet-style pivot table as a DataFrame. It takes arguments for index, columns, and values, and an aggregation function (aggfunc).
```py
data_pivot = {
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-01'],
    'Region': ['East', 'West', 'East', 'West', 'East'],
    'Product': ['A', 'B', 'A', 'C', 'B'],
    'Sales': [100, 150, 120, 200, 180]
}
df_pivot = pd.DataFrame(data_pivot)
print("\nOriginal DataFrame for Pivot:\n", df_pivot)

# Pivot table: total sales by Date and Region
pivot_sales = df_pivot.pivot_table(values='Sales', index='Date', columns='Region', aggfunc='sum')
print("\nPivot Table (Sales by Date and Region):\n", pivot_sales)

# Pivot table: average sales by Product for each Region
pivot_avg_sales = df_pivot.pivot_table(values='Sales', index='Product', columns='Region', aggfunc='mean')
print("\nPivot Table (Average Sales by Product and Region):\n", pivot_avg_sales)
```
## 10. Time Series Functionality (Basic)
Pandas has powerful tools for working with time series data, especially when the DataFrame index is a DatetimeIndex.

### Converting to Datetime:
- `pd.to_datetime()`: Converts arguments to datetime objects.
```py
df_time = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
    'Value': [10, 12, 15, 11]
})
print("\nOriginal Time Series DataFrame:\n", df_time)

df_time['Date'] = pd.to_datetime(df_time['Date'])
print("\nDataFrame dtypes after converting 'Date' to datetime:\n", df_time.dtypes)
```
### Setting Datetime Index:
It's often beneficial to set the datetime column as the DataFrame's index.
```py
df_time = df_time.set_index('Date')
print("\nDataFrame with DatetimeIndex:\n", df_time)
print("\nIndex type:", df_time.index)
```
### Time-based Slicing:
Once the index is a DatetimeIndex, you can easily slice by date.
```py
# Create a longer time series
dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
values = np.random.randint(10, 30, size=10)
df_ts = pd.DataFrame({'Value': values}, index=dates)
print("\nLonger Time Series DataFrame:\n", df_ts)

# Slice by date
print("\nData for '2023-01-05':\n", df_ts.loc['2023-01-05'])
print("\nData for '2023-01-03' to '2023-01-07':\n", df_ts.loc['2023-01-03':'2023-01-07'])
```
### Resampling:
Used for changing the frequency of the time series (e.g., daily to weekly, hourly to daily). Common aggregation methods are `sum()`, `mean()`, `first()`, `last()`.
```py
# Resample to weekly frequency, taking the sum
df_weekly_sum = df_ts.resample('W').sum()
print("\nWeekly sum:\n", df_weekly_sum)

# Resample to monthly frequency, taking the mean
df_monthly_mean = df_ts.resample('M').mean()
print("\nMonthly mean:\n", df_monthly_mean)
```