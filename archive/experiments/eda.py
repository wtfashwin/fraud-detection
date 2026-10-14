import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
...


# Load dataset
df = pd.read_csv('creditcard.csv')

# 1. Missing values
print("🔍 Missing Values:\n", df.isnull().sum())

# 2. Class distribution
class_counts = df['Class'].value_counts()
print("\n📊 Class Distribution:\n", class_counts)

# 3. Visualize class imbalance
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud Transaction Count")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")
plt.savefig('class_distribution.png')
plt.close()

# 4. Distribution of Amount
plt.figure(figsize=(8,4))
sns.histplot(df['Amount'], bins=100, kde=True)
plt.title("Transaction Amount Distribution")
plt.savefig('amount_distribution.png')
plt.close()

# 5. Normalize Amount and Time
from sklearn.preprocessing import StandardScaler

df['scaled_amount'] = StandardScaler().fit_transform(df[['Amount']])
df['scaled_time'] = StandardScaler().fit_transform(df[['Time']])

# Drop original columns
df = df.drop(['Time', 'Amount'], axis=1)

# Save processed dataset
df.to_csv('processed_data.csv', index=False)

print("\n✅ Saved processed dataset as processed_data.csv")
