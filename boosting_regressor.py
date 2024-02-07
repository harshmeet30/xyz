#############################INST###################
#1. Stats of companies automating insurance renewal - mckinsey !

#########   2   ##
# Claim Frequency Analysis:

# Graphs: Histogram, Box plot, Violin plot
# Variables: Policyholder demographics (age, gender, location), Policy type, Coverage amount
# Claim Severity Analysis:

# Graphs: Histogram, Box plot, Violin plot
# Variables: Policyholder demographics, Policy type, Coverage amount
# Temporal Analysis:

# Graphs: Time series plot, Heatmap, Calendar plot
# Variables: Time (year, month), Seasonality indicators, Policyholder demographics
# Policyholder Analysis:

# Graphs: Pie chart, Bar plot, Box plot, Violin plot
# Variables: Age, Gender, Location, Policy type, Coverage amount
# Geospatial Analysis:

# Graphs: Choropleth map, Heatmap
# Variables: Geographic location, Claim frequency, Claim severity, Policyholder demographics
# Feature Engineering:

# Graphs: Histogram, Box plot, Violin plot
# Variables: Policy duration, Lapse history, Coverage type, Deductible levels, Claims history
# Correlation Analysis:

# Graphs: Heatmap, Correlation matrix, Scatter plot
# Variables: Claim frequency vs. Policyholder demographics, Claim severity vs. Policy type, Correlation between engineered features
# Cluster Analysis:

# Graphs: Scatter plot, Dendrogram
# Variables: Cluster membership vs. Policyholder demographics, Cluster centroids vs. Claim frequency/severity
# Model Evaluation:

# Graphs: ROC curve, Precision-Recall curve, Calibration plot
# Variables: Model predictions vs. True outcomes, Model performance metrics vs. Model parameters
# Dashboard or Interactive Visualization:

# Graphs: Interactive plots, Dropdown menus, Filters
# Variables: All relevant variables can be explored dynamically


# 3. Data analysis and some visualizations over some amount of data.

# filters and working of them !
#5. Risk Tolerance directly proportional to tolerance!

#############################INST###################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# Sample data
data = pd.read_csv('your_dataset.csv')  # Load your dataset here

# Separate features and target variable
X = data.drop(columns=['target_column'])  # Specify the target column name
y = data['target_column']

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define GradientBoostingRegressor model
gb_regressor = GradientBoostingRegressor(random_state=42)

# Fit the model to the training data
gb_regressor.fit(X_train, y_train)

# Get feature importances
feature_importances = gb_regressor.feature_importances_

# Sort feature importances in descending order
indices = feature_importances.argsort()[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. Feature '{X.columns[indices[f]]}' - importance: {feature_importances[indices[f]]}")

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), feature_importances[indices], color="b", align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()



###################################################################################
#-------------------------------clustering----------------------------------------


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

# Sample data
data = pd.DataFrame({
    'numeric_col1': [1, 2, 3, 4, 5],
    'numeric_col2': [5, 4, 3, 2, 1],
    'category_col': ['A', 'B', 'C', 'A', 'B']
})

# Define numerical and categorical columns
numeric_cols = ['numeric_col1', 'numeric_col2']
category_cols = ['category_col']

# Pipeline for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), category_cols)
    ])

# Combine preprocessing and clustering
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('cluster', DBSCAN(eps=0.5, min_samples=2))  # Adjust parameters according to your data
])

# Fit the model
pipeline.fit(data)

# Output cluster labels
labels = pipeline.named_steps['cluster'].labels_
print("Cluster labels:", labels)



########################################################