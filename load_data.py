import pandas as pd # type: ignore

# Load CSV
df = pd.read_csv('creditcard.csv')

# Shape
print("📊 Rows, Columns →", df.shape)

# Head
print("\n🔹 Sample rows:")
print(df.head())

# Class distribution
print("\n🔍 Class Distribution:")
print(df['Class'].value_counts())

