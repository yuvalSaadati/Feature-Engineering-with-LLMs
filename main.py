# Import necessary libraries
from caafe import CAAFEClassifier  # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier  # Fast Automated Machine Learning for small tabular datasets
from sklearn.ensemble import RandomForestClassifier  # Alternative classifier (not used in this script)
import torch  # To check if GPU (CUDA) is available for faster computation
from caafe import data  # Utility functions for data loading and preprocessing
from sklearn.metrics import accuracy_score  # To evaluate model performance
from functools import partial  # Used to modify function behavior
import warnings
warnings.filterwarnings("ignore")  # Suppress unnecessary warnings
import logging

# Suppress logs from the DoWhy library to avoid unnecessary output
logging.getLogger("dowhy").setLevel(logging.ERROR)

# Load all available datasets for testing CAAFE
cc_test_datasets_multiclass = data.load_all_data()

# Select a specific dataset (index 8)
dataset_index = 2
ds = cc_test_datasets_multiclass[dataset_index]

# Split dataset into training and testing sets
ds, df_train, df_test, _, _ = data.get_data_split(ds, seed=0)

# Extract the target column name and dataset description
target_column_name = ds[4][-1]
dataset_description = ds[-1]

# Display dataset name (useful for debugging)
ds[dataset_index]  

# Convert dataset features into numeric format (CAAFE requires numerical features)
from caafe.preprocessing import make_datasets_numeric
df_train, df_test = make_datasets_numeric(df_train, df_test, target_column_name)

# Extract features (X) and labels (y) from training and test sets
train_x, train_y = data.get_X_y(df_train, target_column_name)
test_x, test_y = data.get_X_y(df_test, target_column_name)

### Setup Base Classifier (Before Feature Engineering)

# Uncomment to use RandomForestClassifier instead of TabPFNClassifier
# clf_no_feat_eng = RandomForestClassifier()

# # Initialize TabPFNClassifier, using GPU if available for better performance
clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'))

# Ensure that fit() is properly handled using partial (needed for integration with CAAFE)
clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit)

# Train the classifier on the original dataset (before feature engineering)
clf_no_feat_eng.fit(train_x, train_y)

# Make predictions on the test set
pred = clf_no_feat_eng.predict(test_x)

# Calculate accuracy before applying CAAFE
acc = accuracy_score(pred, test_y)
print(f'Accuracy before CAAFE: {acc}')

### Setup and Run CAAFE (Feature Engineering) - Uses OpenAI API

# Initialize the CAAFEClassifier with the base classifier (TabPFNClassifier)
# - Uses GPT-4 as the LLM model
# - Runs for 10 iterations to generate new features
caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                            llm_model="gpt-4",
                            iterations=10)

# Define different experiment types for feature engineering and selection
experiments = ["Feature_Selection", "Causality_Feature_Selection", "Feature_Selection_Causality", "Base"]

# Define different causal analysis methods
causal_methods = ["individual", "dag", "psm"]

# Specify filenames for saving model attributes and dataset snapshots
model_attributes_file = f"models/model_attributes_{dataset_index}.pkl"
df_train_file = f"df/df_train_{dataset_index}.parquet"

# Display the baseline features before CAAFE modifies them
print(f"Baseline Features: {df_train.columns.values}")
dataset_index = 3
# Apply CAAFE to generate new features and enhance the dataset
caafe_clf.fit_pandas(df_train, df_test, causal_methods[0], experiments[dataset_index], model_attributes_file, df_train_file,
                     target_column_name=target_column_name,
                     dataset_description=dataset_description)

# Make predictions using the enhanced dataset after CAAFE
pred = caafe_clf.predict(df_test)

# Calculate accuracy after applying CAAFE
acc = accuracy_score(pred, test_y)
print(f'Accuracy after CAAFE: {acc}')
