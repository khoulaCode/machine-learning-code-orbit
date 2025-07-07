# NumPy for Data Science: A Comprehensive Guide
NumPy (Numerical Python) is a fundamental library for numerical computing in Python. It provides powerful data structures, primarily the ndarray (n-dimensional array), and functions for performing operations on these arrays efficiently. It's the backbone of many other scientific computing libraries like Pandas, SciPy, and Scikit-learn.
## 1. Array Creation
NumPy arrays can be created in various ways:
### 1.1 From Python Lists
The most common way to create a NumPy array is from a standard Python list or a list of lists.
import numpy as np

```python
# From a 1D list
list_1d = [1, 2, 3, 4, 5]
arr_1d = np.array(list_1d)
print("1D Array:", arr_1d)
print("Type:", type(arr_1d))
print("Shape:", arr_1d.shape)
print("Dimensions (ndim):", arr_1d.ndim)
print("Data Type (dtype):", arr_1d.dtype)

# From a 2D list (list of lists)
list_2d = [[1, 2, 3], [4, 5, 6]]
arr_2d = np.array(list_2d)
print("\n2D Array:\n", arr_2d)
print("Shape:", arr_2d.shape) # (rows, columns)
print("Dimensions (ndim):", arr_2d.ndim)
```

### 1.2 Initializing Arrays with Placeholder Values
NumPy provides functions to create arrays filled with specific values (like zeros or ones) or an empty array.

`np.zeros(shape, dtype=float)`: Creates an array filled with zeros.\
`np.ones(shape, dtype=float)`: Creates an array filled with ones.\
`np.full(shape, fill_value, dtype=None)`: Creates an array filled with a specified value.\
`np.empty(shape, dtype=float)`: Creates an array without initializing its entries. Their initial content is random and depends on the state of the memory.
```python
# Array of zeros
zeros_arr = np.zeros((3, 4)) # 3 rows, 4 columns
print("\nZeros Array:\n", zeros_arr)

# Array of ones
ones_arr = np.ones((2, 3)) # 2 rows, 3 columns
print("\nOnes Array:\n", ones_arr)

# Array filled with a specific value
full_arr = np.full((2, 2), 7)
print("\nFull Array (filled with 7s):\n", full_arr)

# Empty array (values are uninitialized)
empty_arr = np.empty((2, 2))
print("\nEmpty Array (uninitialized values):\n", empty_arr)
```

### 1.3 Creating Arrays with a Range of Numbers
Similar to Python's `range()` function, NumPy offers `arange()` and `linspace()` for generating sequences.\
`np.arange(start, stop, step)`: Returns evenly spaced values within a given interval. Similar to `range()`.\
`np.linspace(start, stop, num)`: Returns evenly spaced numbers over a specified interval.
```python
# Using arange
range_arr = np.arange(0, 10, 2) # Start, Stop (exclusive), Step
print("\nArange Array (0 to 10, step 2):", range_arr)

# Using linspace
linspace_arr = np.linspace(0, 10, 5) # Start, Stop (inclusive), Number of elements
print("Linspace Array (5 elements from 0 to 10):", linspace_arr)
```

### 1.4 Random Arrays
NumPy's random module is essential for creating arrays with random values, often used in simulations and machine learning.
np.random.rand(d0, d1, ..., dn): Creates an array of the given shape with random samples from a uniform distribution over [0,1).
np.random.randn(d0, d1, ..., dn): Creates an array of the given shape with random samples from a standard normal distribution (mean 0, variance 1).
np.random.randint(low, high, size): Creates an array of random integers from low (inclusive) to high (exclusive).
```python
# Random floats between 0 and 1
rand_arr = np.random.rand(3, 3)
print("\nRandom Array (uniform distribution):\n", rand_arr)

# Random floats from standard normal distribution
randn_arr = np.random.randn(2, 2)
print("\nRandom Array (normal distribution):\n", randn_arr)

# Random integers
randint_arr = np.random.randint(0, 10, size=(2, 4)) # Integers from 0 to 9
print("\nRandom Integers Array (0 to 9):\n", randint_arr)
```

## 2. Python Lists vs. NumPy Arrays: Speed and Efficiency
This is one of the most crucial aspects of understanding why NumPy is preferred for numerical operations in data science.
### 2.1 Speed Comparison
NumPy operations are significantly faster than equivalent operations on Python lists, especially for large datasets. This is because NumPy arrays are implemented in C and Fortran, allowing for highly optimized, pre-compiled code execution. Python lists, on the other hand, are lists of pointers to Python objects, and operations on them involve Python's slower interpreter.
Let's illustrate with an example: element-wise addition.
import time
```python
# Define the size of the arrays
size = 1_000_000

# Python lists
list1 = list(range(size))
list2 = list(range(size))

start_time = time.time()
result_list = [x + y for x, y in zip(list1, list2)]
end_time = time.time()
print(f"\nTime taken for Python lists addition: {end_time - start_time:.6f} seconds")

# NumPy arrays
np_arr1 = np.arange(size)
np_arr2 = np.arange(size)

start_time = time.time()
result_numpy = np_arr1 + np_arr2
end_time = time.time()
print(f"Time taken for NumPy arrays addition: {end_time - start_time:.6f} seconds")
```

You'll observe that NumPy's execution time is orders of magnitude faster.
### 2.2 Efficiency (Memory Usage)

NumPy arrays are also more memory-efficient.

**Python Lists:** Each element in a Python list is a separate Python object, and each object carries its own overhead (type information, reference count, etc.). This means a list of integers doesn't store just the integers; it stores pointers to integer objects, which are spread out in memory.

**NumPy Arrays:** NumPy arrays store elements of the same data type contiguously in memory. This contiguous block of memory is much more efficient for CPU cache utilization and vectorization, further contributing to speed gains.

**Consider storing a million integers:**
A Python list would store a million pointers, each pointing to an integer object.
A NumPy array would store a million integers directly in a contiguous block of memory.
import sys

```python
# Python list of integers
list_of_ints = list(range(1000))
size_of_list_elements = sum(sys.getsizeof(i) for i in list_of_ints)
print(f"\nSize of 1000 Python integers (elements only): {size_of_list_elements} bytes")
print(f"Total size of Python list object: {sys.getsizeof(list_of_ints)} bytes")
print(f"Average size per element (including overhead): {sys.getsizeof(list_of_ints) / len(list_of_ints):.2f} bytes")

# NumPy array of integers (e.g., int64)
numpy_array_of_ints = np.arange(1000, dtype=np.int64) # Explicitly use int64 for comparison
print(f"\nSize of NumPy array (1000 int64 elements): {numpy_array_of_ints.nbytes} bytes")
print(f"Total size of NumPy array object (including metadata): {sys.getsizeof(numpy_array_of_ints)} bytes")
print(f"Average size per element (NumPy, 64-bit int): {numpy_array_of_ints.nbytes / len(numpy_array_of_ints):.2f} bytes")
```

You'll see that the NumPy array uses significantly less memory per element. The nbytes attribute gives the total bytes consumed by the array data.
### 2.3 Functionality
NumPy arrays come with a vast collection of built-in functions for linear algebra, Fourier transforms, random number generation, and more, which are not available for standard Python lists. This rich functionality makes complex numerical computations much easier and more efficient.
## 3. Array Manipulation
Manipulating arrays is a core task in data analysis.
### 3.1 Reshaping
Changing the shape of an array without changing its data. The new shape must have the same total number of elements.
```array.reshape(new_shape):``` Returns a new array with the specified shape.
```array.ravel()``` or ```array.flatten()```: Flattens the array into a 1D ```array. ravel() ``` returns a view when possible, flatten() always returns a copy.
```python
arr = np.arange(1, 10) # [1 2 3 4 5 6 7 8 9]
print("\nOriginal array:", arr)

# Reshape to 3x3
reshaped_arr = arr.reshape((3, 3))
print("\nReshaped to 3x3:\n", reshaped_arr)

# Flatten back to 1D
flattened_arr = reshaped_arr.flatten()
print("\nFlattened array:", flattened_arr)
```

### 3.2 Transposing
Changing rows into columns and columns into rows.
```array.T``` or ```np.transpose(array)```
```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\nOriginal 2D array:\n", arr_2d)

transposed_arr = arr_2d.T
print("\nTransposed array:\n", transposed_arr)
```

### 3.3 Stacking (Concatenation)
Combining multiple arrays along an existing or new axis.
```np.vstack((arr1, arr2, ...))```: Stacks arrays vertically (row-wise).
```np.hstack((arr1, arr2, ...))```: Stacks arrays horizontally (column-wise).
```np.concatenate((arr1, arr2, ...), axis=...)```: General function for stacking along any axis.
```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Vertical stack
v_stack = np.vstack((arr1, arr2))
print("\nVertical Stack:\n", v_stack)

# Horizontal stack
h_stack = np.hstack((arr1, arr2))
print("\nHorizontal Stack:\n", h_stack)

# Concatenate along axis 0 (rows) - equivalent to vstack for 2D
concat_axis0 = np.concatenate((arr1, arr2), axis=0)
print("\nConcatenate Axis 0:\n", concat_axis0)

# Concatenate along axis 1 (columns) - equivalent to hstack for 2D
concat_axis1 = np.concatenate((arr1, arr2), axis=1)
print("\nConcatenate Axis 1:\n", concat_axis1)
```

### 3.4 Splitting
Dividing an array into multiple smaller arrays.
`np.vsplit(array, indices_or_sections)`: Splits an array into multiple sub-arrays vertically (row-wise).
`np.hsplit(array, indices_or_sections)`: Splits an array into multiple sub-arrays horizontally (column-wise).
`np.split(array, indices_or_sections, axis=...)`: General function for splitting along any axis.
```python
arr = np.arange(16).reshape((4, 4))
print("\nOriginal array for splitting:\n", arr)

# Split into 2 equal parts vertically
v_split = np.vsplit(arr, 2)
print("\nVertical Split (2 parts):\n", v_split[0], "\n---\n", v_split[1])

# Split into 4 equal parts horizontally
h_split = np.hsplit(arr, 4)
print("\nHorizontal Split (4 parts):")
for part in h_split:
    print(part)

# Split along axis 1 at specific indices
split_at_indices = np.split(arr, [1, 3], axis=1) # Split before column 1, and before column 3
print("\nSplit at indices [1, 3] along axis 1:")
for part in split_at_indices:
    print(part)
```

### 3.5 Indexing and Slicing

Accessing parts of an array, similar to Python lists but with N-dimensional capabilities.
Basic Indexing (1D):
```py
arr_1d = np.array([10, 20, 30, 40, 50])
print("\n1D array:", arr_1d)
print("Element at index 2:", arr_1d[2])       # Output: 30
print("Last element:", arr_1d[-1])           # Output: 50
```

Basic Indexing (2D): array[row_index, column_index]
```py
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D array:\n", arr_2d)
print("Element at (0, 1):", arr_2d[0, 1])     # Output: 2
print("Element at (2, 0):", arr_2d[2, 0])     # Output: 7
```

Slicing (1D): array[start:stop:step]
```py
arr_1d = np.array([10, 20, 30, 40, 50, 60])
print("\n1D array for slicing:", arr_1d)
print("First three elements:", arr_1d[0:3])    # Output: [10 20 30]
print("Elements from index 2 onwards:", arr_1d[2:])   # Output: [30 40 50 60]
print("Every second element:", arr_1d[::2])   # Output: [10 30 50]
```

Slicing (2D): array[row_slice, column_slice]
```py
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
print("\n2D array for slicing:\n", arr_2d)
print("First row:", arr_2d[0, :])             # Output: [1 2 3 4]
print("First two rows, first two columns:\n", arr_2d[0:2, 0:2])
print("All rows, second column:", arr_2d[:, 1]) # Output: [2 6 10]
```

Boolean Indexing: Selecting elements based on a boolean condition.
```py
arr = np.array([10, 5, 20, 15, 30])
print("\nArray for boolean indexing:", arr)
mask = (arr > 10)
print("Boolean mask (elements > 10):", mask)
print("Elements greater than 10:", arr[mask]) # Output: [20 15 30]

# Combined operation
print("Even numbers:", arr[arr % 2 == 0])
```

Fancy Indexing: Selecting elements at specific, non-contiguous indices.
```py
arr = np.array([10, 20, 30, 40, 50, 60])
print("\nArray for fancy indexing:", arr)
indices = [0, 3, 5]
print("Elements at specific indices [0, 3, 5]:", arr[indices]) # Output: [10 40 60]

# For 2D arrays:
arr_2d = np.array([[10, 20, 30],
                   [40, 50, 60],
                   [70, 80, 90]])
print("\n2D array for fancy indexing:\n", arr_2d)
rows = np.array([0, 2, 0])
cols = np.array([0, 1, 2])
print("Elements at (0,0), (2,1), (0,2):", arr_2d[rows, cols]) # Output: [10 80 30]
```
Note: Fancy indexing returns a copy of the data, whereas slicing often returns a view.
## 4. Numerical Operations
NumPy excels at performing fast numerical operations.
### 4.1 Basic Arithmetic Operations
Element-wise operations are performed directly.
```py
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print("\nArray 1:", arr1)
print("Array 2:", arr2)

print("Addition (arr1 + arr2):", arr1 + arr2)
print("Subtraction (arr1 - arr2):", arr1 - arr2)
print("Multiplication (arr1 * arr2):", arr1 * arr2) # Element-wise multiplication
print("Division (arr1 / arr2):", arr1 / arr2)
print("Exponentiation (arr1 ** 2):", arr1 ** 2)
```

### 4.2 Universal Functions (ufuncs)
NumPy provides "universal functions" (ufuncs) that operate element-by-element on arrays. These are highly optimized C functions.
Examples include `np.sqrt()`, `np.exp()`, `np.log()`, `np.sin()`, `np.cos()`, `np.abs()`, `np.maximum()`, etc.
```py
arr = np.array([1, 4, 9, 16])
print("\nArray for ufuncs:", arr)

print("Square root:", np.sqrt(arr))
print("Exponential:", np.exp(arr))

arr_a = np.array([1, 5, 2])
arr_b = np.array([4, 3, 6])
print("Maximum (element-wise):", np.maximum(arr_a, arr_b))
```

### 4.3 Aggregation Functions
Functions that summarize the data in an array (e.g., sum, mean, min, max, std). They can be applied to the entire array or along a specific axis.
`np.sum()`, `np.mean()`, `np.min()`, `np.max()`, `np.std()`, `np.var()`, `np.median()`
```py
arr_agg = np.array([[1, 2, 3],
                    [4, 5, 6]])
print("\nArray for aggregation:\n", arr_agg)

print("Sum of all elements:", np.sum(arr_agg))
print("Mean of all elements:", np.mean(arr_agg))
print("Minimum of all elements:", np.min(arr_agg))
print("Maximum of all elements:", np.max(arr_agg))

# Aggregation along an axis
print("\nSum along axis 0 (columns sum up to rows):", np.sum(arr_agg, axis=0)) # Sum of each column
print("Mean along axis 1 (rows mean):", np.mean(arr_agg, axis=1))     # Mean of each row
```

`axis=0` refers to operations along the columns (collapsing rows).
`axis=1` refers to operations along the rows (collapsing columns).
### 4.4 Broadcasting
Broadcasting is a powerful mechanism that allows NumPy to work with arrays of different shapes when performing arithmetic operations. It avoids the need for explicit loops, making operations more efficient.

The smaller array is "broadcast" across the larger array so that they have compatible shapes. Broadcasting rules:

If the arrays do not have the same number of dimensions, the shape of the smaller array is padded with ones on its left side.

If the shape of the two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.
If in any dimension, the sizes disagree and neither is 1, an error is raised.
```py
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
scalar = 10
print("\nArray for broadcasting:\n", arr)
print("Scalar:", scalar)

# Add scalar to every element
result_scalar_add = arr + scalar
print("\nArray + Scalar:\n", result_scalar_add)

vector = np.array([100, 200, 300]) # A 1D array of shape (3,)
print("\nVector for broadcasting:", vector)

# Add vector to each row of the 2D array
# The vector (1,3) is broadcast to (3,3) by copying itself downwards.
result_vector_add = arr + vector
print("\nArray + Vector (broadcast row-wise):\n", result_vector_add)

# Example: Adding a column vector
col_vector = np.array([[10], [20], [30]]) # A 2D array of shape (3,1)
print("\nColumn vector:\n", col_vector)

# The column vector (3,1) is broadcast to (3,3) by copying itself across columns.
result_col_vector_add = arr + col_vector
print("\nArray + Column Vector (broadcast column-wise):\n", result_col_vector_add)
```

Broadcasting is a flexible and powerful tool, but it can sometimes lead to unexpected results if the shapes are not carefully considered. Always ensure your array dimensions are compatible for the intended operation.

## 5. Masking and Filtering
Masking and filtering in NumPy allow you to select and work with subsets of data based on specific conditions. This is useful for extracting relevant data, performing conditional operations, and analyzing subsets of data. You can perform filtering by creating a Boolean array (mask) where each element indicates whether the corresponding element in the original array meets the specified condition. This mask is then used to index the original array, extracting the elements that satisfy the condition.

### 5.1 Basic Filtering with Boolean Indexing
Boolean indexing allows you to filter array elements based on conditions.

```Python

import numpy as np

# Create an array
array = np.array([1, 5, 8, 12, 20, 3])
print("Original Array:", array)

# Define a condition
condition = array > 10
print("Boolean Mask (array > 10):", condition)

# Apply the condition to filter the array
filtered_array = array[condition]
print("Filtered Array (elements > 10):", filtered_array)
```
### 5.2 Filtering with Multiple Conditions
You can combine multiple Boolean conditions using logical operators (`&` for AND, `|` for OR, `~` for NOT).

```Python

# Filter elements within a range using multiple conditions
array = np.array([1, 5, 8, 12, 20, 3])
print("\nOriginal Array:", array)

condition_multiple = (array > 5) & (array < 15)
print("Boolean Mask (5 < elements < 15):", condition_multiple)

filtered_array_multiple = array[condition_multiple]
print("Filtered Array (5 < elements < 15):", filtered_array_multiple)
```
### 5.3 Filtering with `np.where()`
The `np.where()` function returns the indices where a condition is `True`. These indices can then be used to extract the filtered elements.

```Python

# Filtering using np.where() function
array = np.array([1, 5, 8, 12, 20, 3])
print("\nOriginal Array:", array)

condition_where = array > 10
filtered_indices = np.where(condition_where)
filtered_array_where = array[filtered_indices]
print("Filtered Array (elements > 10) using np.where:", filtered_array_where)
```
### 5.4 Masked Arrays
NumPy's `numpy.ma` module provides a way to handle missing or invalid entries in arrays. A masked array is a combination of a standard `numpy.ndarray` and a Boolean mask. When an element of the mask is `True`, the corresponding element in the data array is considered masked (invalid). This allows computations to skip or flag unwanted entries without deleting them.

```Python

import numpy.ma as ma

x = np.array([1, 2, 3, -1, 5])
# Mask the fourth entry as invalid (index 3)
mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0])
print("\nOriginal array with invalid entry (-1):", x)
print("Masked Array (value at index 3 is masked):", mx)
print("Mean of masked array (skips masked value):", mx.mean())

# Accessing valid and invalid elements
print("Mask of the masked array:", mx.mask)
valid_elements = mx[~mx.mask] # ~ negates the boolean mask
print("Valid elements:", valid_elements)
```
