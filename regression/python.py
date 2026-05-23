from common_imports import *
from python import *

# df is modified in upper
df_analysis_regression = df_analysis.copy()

# df_analysis_regression.head(5)
df_analysis_regression['TotalCharges'].dropna()

corelation of numeric only

# Calculate the correlation matrix for the regression analysis
corr_mat_reg = df_analysis_regression.select_dtypes(include=[np.number]).corr()

# Display the correlation matrix, focusing on a potential regression target like 'MonthlyCharges'
plt.figure(figsize=(14, 10))
sns.heatmap(corr_mat_reg, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix for Regression Analysis')
plt.show()

print("Correlation of features with MonthlyCharges (potential regression target):")
print(corr_mat_reg['MonthlyCharges'].sort_values(ascending=False))

### Correlation including non-numeric features (after encoding)

from sklearn.preprocessing import LabelEncoder

df_encoded_for_corr = df_analysis_regression.copy()

# Drop 'customerID' as it's just an identifier and not useful for correlation
df_encoded_for_corr = df_encoded_for_corr.drop('customerID', axis=1)

# Identify categorical columns that are not yet encoded
categorical_cols = df_encoded_for_corr.select_dtypes(include='object').columns

# Apply Label Encoding to remaining categorical columns
le = LabelEncoder()
for col in categorical_cols:
    df_encoded_for_corr[col] = le.fit_transform(df_encoded_for_corr[col])

print("DataFrame with all features numerically encoded:")
display(df_encoded_for_corr.head())


# Calculate the full correlation matrix on the encoded DataFrame
full_corr_matrix = df_encoded_for_corr.corr()

# Plot the heatmap for the full correlation matrix
plt.figure(figsize=(20, 15)) # Adjust figure size for more features
sns.heatmap(full_corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Full Feature Correlation Matrix (All features encoded)')
plt.show()

# Optional: Display correlation with MonthlyCharges for the full set of features
print("Correlation of all encoded features with MonthlyCharges:")
print(full_corr_matrix['MonthlyCharges'].sort_values(ascending=False))

full_corr_matrix.columns


selected_col_reg = ['TotalCharges', 'MultipleLines', 'PaperlessBilling', 'StreamingMovies', 'StreamingTV', 'tenure']

scaler_reg = StandardScaler()
df_encoded_for_corr['tenure'] = scaler_reg.fit_transform(df_encoded_for_corr[['tenure']])
df_encoded_for_corr['TotalCharges'] = scaler_reg.fit_transform(df_encoded_for_corr[['TotalCharges']])

# selected_col_reg = ['TotalCharges', 'MultipleLines', 'PaperlessBilling', 'StreamingMovies', 'StreamingTV', 'tenure']

data_pre_reg = df_encoded_for_corr[selected_col_reg].copy()

data_pre_reg.head(5)

data_pre_reg = df_encoded_for_corr[selected_col_reg].copy()
# Combine features and target to drop NaNs consistently
temp_df_reg = df_encoded_for_corr[selected_col_reg + ['MonthlyCharges']].copy()
temp_df_reg.dropna(inplace=True)

x_reg = temp_df_reg[selected_col_reg]
y_reg = temp_df_reg['MonthlyCharges']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg, random_state=42)

avg = np.mean(df_encoded_for_corr['TotalCharges'])
X_train_reg,y_train_reg.fillna(avg)