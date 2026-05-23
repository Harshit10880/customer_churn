from common_imports import *

df = pd.read_csv('customer.csv')


df.head(5)

df.info()

df.isnull().sum()

df.shape

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

sns.set_theme(style="whitegrid")

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {col}')


# check co-relation between features

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# We use a copy so we don't mess up the original dataframe yet
df_analysis = df.copy()
df_analysis['Churn'] = df_analysis['Churn'].map({'Yes': 1, 'No': 0})
df_analysis['gender'] = df_analysis['gender'].map({'Male': 1, 'Female': 0})
df_analysis['Partner'] = df_analysis['Partner'].map({'Yes': 1, 'No': 0})
df_analysis['Dependents'] = df_analysis['Dependents'].map({'Yes': 1, 'No': 0})

# We only take the numeric columns
corr_matrix = df_analysis.select_dtypes(include=[np.number]).corr()

# code for heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation with Churn')
plt.show()

print("Correlation of features with the Target (Churn):")
print(corr_matrix['Churn'].sort_values(ascending=False))

# data preprocessing

# selected_columns = ['tenure', 'TotalCharges', 'SeniorCitizen', 'MonthlyCharges', 'Churn']
selected_columns = ['tenure', 'TotalCharges', 'SeniorCitizen', 'MonthlyCharges', 'Churn', 'Dependents', 'Partner']
datapre = df[selected_columns].copy()

datapre.head(5)

le = LabelEncoder()

# Apply LabelEncoder to 'Dependents' and 'Partner'
for col in ['Dependents', 'Partner']:
    if col in datapre.columns:
        datapre[col] = le.fit_transform(datapre[col])

# Ensure 'Churn' is also encoded (if it's not already numeric)
datapre['Churn'] = le.fit_transform(datapre['Churn'])

datapre

scaler = StandardScaler()
datapre[['MonthlyCharges', 'TotalCharges', 'tenure']] = scaler.fit_transform(datapre[['MonthlyCharges', 'TotalCharges', 'tenure']])

print("Preprocessed Data for Training:")
print(datapre.head())
print("\nFeature Correlation with Churn in this subset:")
print(datapre.corr()['Churn'])

# data spliting

# Remove rows with NaN values before splitting data
datapre.dropna(inplace=True)

X = datapre.drop('Churn', axis=1) # The 4 columns
y = datapre['Churn']              # The target

# 2. Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 3. Initialize and Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)