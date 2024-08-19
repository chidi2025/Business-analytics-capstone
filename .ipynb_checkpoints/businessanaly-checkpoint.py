import pandas as pd
# Plot distributions, correlations, etc.
import seaborn as sns
import matplotlib.pyplot as plt

# File path
file_path = 'data/walmart_sales_data.csv'

# Load the CSV file
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print(f"File not found. Please check the path: {file_path}")


# Identify non-numeric columns
non_numeric_columns = df.select_dtypes(include=['object']).columns
print(non_numeric_columns)

# Drop non-numeric columns or encode them
df_numeric = df.drop(non_numeric_columns, axis=1)

# Calculate the correlation matrix
correlation_matrix = df_numeric.corr()

# Visualize the correlation matrix (Optional)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Display the first few rows of the dataframe
df.head()

# Check for missing values
missing_values = df.isnull().sum()

# Drop rows with missing values or handle them accordingly
df = df.dropna()  # or use df.fillna(value)

# Drop the non-numeric columns 'Date' and 'Weather'
df_numeric = df.drop(['Date', 'Weather'], axis=1)

# Now calculate the correlation matrix
correlation_matrix = df_numeric.corr()

# Display the correlation matrix 
print(correlation_matrix)

# Visualize the correlation matrix (Optional)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Convert columns to appropriate data types if needed
df['Date'] = pd.to_datetime(df['Date'])

# Remove duplicate rows if any
df = df.drop_duplicates()


# Creating new features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

# Statistical summary
df.describe()

# Correlation matrix
correlation_matrix = df.corr()



# Correlation heatmap
sns.heatmap(correlation_matrix, annot=True)
plt.show()

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

# Assume 'Sales' is the target variable and others are features
X = df.drop('Sales', axis=1)
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error: {rmse}')
