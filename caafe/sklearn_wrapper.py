from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from .run_llm_code import run_llm_code
from .preprocessing import (
    make_datasets_numeric,
    split_target_column,
    make_dataset_numeric,
)
from caafe import data
from .data import get_X_y
from .caafe import generate_features
from .metrics import auc_metric, accuracy_metric
import pandas as pd
import numpy as np
from typing import Optional
import pandas as pd
import os
from sklearn.utils.multiclass import unique_labels
import pickle
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from dowhy import CausalModel
import re
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

class CAAFEClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses the CAAFE algorithm to generate features and a base classifier to make predictions.

    Parameters:
    base_classifier (object, optional): The base classifier to use. If None, a default TabPFNClassifier will be used. Defaults to None.
    optimization_metric (str, optional): The metric to optimize during feature generation. Can be 'accuracy' or 'auc'. Defaults to 'accuracy'.
    iterations (int, optional): The number of iterations to run the CAAFE algorithm. Defaults to 10.
    llm_model (str, optional): The LLM model to use for generating features. Defaults to 'gpt-3.5-turbo'.
    n_splits (int, optional): The number of cross-validation splits to use during feature generation. Defaults to 10.
    n_repeats (int, optional): The number of times to repeat the cross-validation during feature generation. Defaults to 2.
    """
    def __init__(
        self,
        base_classifier: Optional[object] = None,
        optimization_metric: str = "accuracy",
        iterations: int = 10,
        llm_model: str = "gpt-3.5-turbo",
        n_splits: int = 10,
        n_repeats: int = 2,
    ) -> None:
        self.base_classifier = base_classifier
        if self.base_classifier is None:
            from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
            import torch
            from functools import partial

            self.base_classifier = TabPFNClassifier(
                N_ensemble_configurations=16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.base_classifier.fit = partial(
                self.base_classifier.fit, overwrite_warning=True
            )
        self.llm_model = llm_model
        self.iterations = iterations
        self.optimization_metric = optimization_metric
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def fit_pandas(self, df,df_test, causal_method,experiment, model_attributes_file,df_train_file, dataset_description, target_column_name, **kwargs):
        """
        Fit the classifier to a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame to fit the classifier to.
        dataset_description (str): A description of the dataset.
        target_column_name (str): The name of the target column in the DataFrame.
        **kwargs: Additional keyword arguments to pass to the base classifier's fit method.
        """
        feature_columns = list(df.drop(columns=[target_column_name]).columns)

        X, y = (
            df.drop(columns=[target_column_name]).values,
            df[target_column_name].values,
        )
        if experiment == "Feature_Selection":
            return self.fit_feature_selection(
                X, y,df_test,model_attributes_file,df_train_file, dataset_description, feature_columns, target_column_name, **kwargs
            )
        elif experiment == "Causality_Feature_Selection":
            return self.fit_causility_feature_selection(
                X, y,df_test,model_attributes_file,df_train_file, causal_method, dataset_description, feature_columns, target_column_name, **kwargs
            )
        elif experiment == "Feature_Selection_Causality":
            return self.fit_feature_selection_causality(
                X, y,df_test,model_attributes_file,df_train_file, causal_method, dataset_description, feature_columns, target_column_name, **kwargs
            )
        else:
            return self.fit(
                X, y,df_test, model_attributes_file,df_train_file, dataset_description, feature_columns, target_column_name, **kwargs
            )


    def fit(
        self, X, y, df_test,model_attributes_file,df_train_file, dataset_description, feature_names, target_name, disable_caafe=False
    ):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.
        dataset_description : str
            A description of the dataset.
        feature_names : List[str]
            The names of the features in the dataset.
        target_name : str
            The name of the target variable in the dataset.
        disable_caafe : bool, optional
            Whether to disable the CAAFE algorithm, by default False.
        """
        self.dataset_description = dataset_description
        self.feature_names = list(feature_names)
        self.target_name = target_name

        self.X_ = X
        self.y_ = y

        if X.shape[0] > 3000 and self.base_classifier.__class__.__name__ == "TabPFNClassifier":
            print(
                "WARNING: TabPFN may take a long time to run on large datasets. Consider using alternatives (e.g. RandomForestClassifier)"
            )
        elif X.shape[0] > 10000 and self.base_classifier.__class__.__name__ == "TabPFNClassifier":
            print("WARNING: CAAFE may take a long time to run on large datasets.")

        ds = [
            "dataset",
            X,
            y,
            [],
            self.feature_names + [target_name],
            {},
            dataset_description,
        ]
        
        # Load cached data if available
        if os.path.exists(df_train_file) and os.path.exists(model_attributes_file):
            print("Loading pre-processed dataset and attributes from files...")
            df_train = pd.read_parquet(df_train_file)
            with open(model_attributes_file, 'rb') as f:
                self.code, self.mappings= pickle.load(f)
            df_train = self.remove_duplicate_columns(df_train)
            # df_train, _ = remove_nan_values(df_train, None)

            self.feature_names = [col for col in df_train.columns if col != target_name]
            print(f"Generated Features (All): {self.feature_names}")

        else:
            print("Processing dataset and saving for future use...")

            # Add X and y as one dataframe
            df_train = pd.DataFrame(
                X,
                columns=self.feature_names,
            )
            df_train[target_name] = y
            if disable_caafe:
                self.code = ""
            else:
                self.code, prompt, messages = generate_features(
                    ds,
                    df_train,
                    model=self.llm_model,
                    iterative=self.iterations,
                    metric_used=auc_metric,
                    iterative_method=self.base_classifier,
                    display_method="markdown",
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                )

            df_train = run_llm_code(
                self.code,
                df_train,
            )
            print(df_train.columns.values)
                   
            df_train, _, self.mappings = make_datasets_numeric(
                df_train, df_test=None, target_column=target_name, return_mappings=True
            )
            # Save processed data and attributes
            df_train.to_parquet(df_train_file, index=False)
            with open(model_attributes_file, 'wb') as f:
                pickle.dump((self.code, self.mappings), f)
            print("Processed dataset and attributes saved.")
             
        df_train, y = split_target_column(df_train, target_name)

        X, y = df_train.values, y.values.astype(int)
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.base_classifier.fit(X, y)

        # Return the classifier
        return self


    # Store feature names after new code execution
    def standardize_column_name(self, col_name):
        """Converts column names to lowercase and removes special characters like underscores and dashes."""
        return re.sub(r'[^a-zA-Z0-9]', '', col_name.lower())

    def remove_duplicate_columns(self, df):
        """Removes duplicate columns that differ only by capitalization or special characters."""
        standardized_cols = {}
        cols_to_drop = []

        for col in df.columns:
            std_col = self.standardize_column_name(col)
            if std_col in standardized_cols:
                cols_to_drop.append(col)  # Mark this column for removal
            else:
                standardized_cols[std_col] = col  # Store the first occurrence

        return df.drop(columns=cols_to_drop)
    # # only feature selection:
    def fit_feature_selection(self, X, y, df_test, model_attributes_file,df_train_file, dataset_description, feature_names, target_name, disable_caafe=False):
            """
            Fit the model to the training data and save intermediate processed data.

            Parameters:
            -----------
            X : np.ndarray
                The training data features.
            y : np.ndarray
                The training data target values.
            dataset_description : str
                A description of the dataset.
            feature_names : List[str]
                The names of the features in the dataset.
            target_name : str
                The name of the target variable in the dataset.
            disable_caafe : bool, optional
                Whether to disable the CAAFE algorithm, by default False.
            """
            self.dataset_description = dataset_description
            self.feature_names = list(feature_names)
            self.target_name = target_name
            

            self.X_ = X
            self.y_ = y

            # Define file paths for caching processed data and attributes

            processed_file = df_train_file
            attributes_file = model_attributes_file
            # Load cached data if available
            if os.path.exists(processed_file) and os.path.exists(attributes_file):
                print("Loading pre-processed dataset and attributes from files...")
                df_train = pd.read_parquet(processed_file)
                with open(attributes_file, 'rb') as f:
                    self.code, self.mappings = pickle.load(f)
                df_train = self.remove_duplicate_columns(df_train)

                self.feature_names = [col for col in df_train.columns if col != target_name]
                print(f"Generated Features (All): {self.feature_names}")

            else:
                print("Processing dataset and saving for future use...")

                # Create initial dataframe
                df_train = pd.DataFrame(X, columns=self.feature_names)
                df_train[target_name] = y

                if disable_caafe:
                    self.code = ""
                else:
                    self.code, prompt, messages = generate_features(
                        ["dataset", X, y, [], self.feature_names + [target_name], {}, dataset_description],
                        df_train,
                        model=self.llm_model,
                        iterative=self.iterations,
                        metric_used=auc_metric,
                        iterative_method=self.base_classifier,
                        display_method="markdown",
                        n_splits=self.n_splits,
                        n_repeats=self.n_repeats,
                    )

                df_train = run_llm_code(self.code, df_train)

                # Convert dataset to numeric format
                df_train, _, self.mappings = make_datasets_numeric(
                    df_train, df_test=None, target_column=target_name, return_mappings=True
                )

                # Save processed data and attributes
                df_train.to_parquet(processed_file, index=False)
                with open(attributes_file, 'wb') as f:
                    pickle.dump((self.code, self.mappings), f)
                print("Processed dataset and attributes saved.")
                    # Store feature names after new code execution
                    
        
            # Feature selection experiments
            feature_selection_methods = {
                "RFE": RFE(RandomForestClassifier(), n_features_to_select=10),
                "LassoCV": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0)),
                "ElasticNetCV": make_pipeline(StandardScaler(), ElasticNetCV(cv=5, random_state=0)),
                "Mutual Information": mutual_info_classif,
                "Boruta": BorutaPy(RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5), n_estimators='auto', random_state=0),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=0),
                "Decision Tree": DecisionTreeClassifier(random_state=0)
            }
            train_x, train_y = split_target_column(df_train, target_name)
            test_x, test_y = split_target_column(df_test, target_name)

            # train_x, train_y = data.get_X_y(df_train, target_name)
            # test_x, test_y  = data.get_X_y(df_test, target_name)

            test_x = run_llm_code(self.code, test_x)
            test_x, _, _ = make_datasets_numeric(
                test_x, df_test=None, target_column=target_name, return_mappings=True
            )
            train_x, test_x = remove_nan_values(train_x, test_x)

            best_selected_features = []
            best_score = -float("inf")
            for method_name, method in feature_selection_methods.items():
                print("")
                print(f"Applying feature selection using {method_name}...")
                if method_name in ["RFE"]:
                    method.fit(train_x, train_y)
                            # Get the selected features
                    selected_features = train_x.columns[method.support_].tolist()
                elif method_name in ["LassoCV"]:
                    method.fit(train_x, train_y)
                            # Get the selected features
                    selected_features = [
                        feature
                        for feature, coef in zip(self.feature_names, method.named_steps['lassocv'].coef_)
                        if coef != 0
                        ]
                elif method_name in ["ElasticNetCV"]:
                    method.fit(train_x, train_y)
                    selected_features = [feature for feature, coef in zip(self.feature_names, method.named_steps['elasticnetcv'].coef_) if coef != 0]
                elif method_name in ["Mutual Information"]:
                    scores = method(train_x, train_y)
                    threshold = np.median(scores)
                    selected_features = [feature for feature, score in zip(self.feature_names, scores) if score > threshold]
                elif method_name in ["Boruta"]:
                    method.fit(train_x, train_y)
                    selected_features = [feature for feature, support in zip(self.feature_names, method.support_) if support]
                elif method_name in ["Gradient Boosting", "Decision Tree"]:
                    method.fit(train_x, train_y)
                    importances = method.feature_importances_
                    threshold = np.mean(importances)
                    selected_features = [feature for feature, importance in zip(self.feature_names, importances) if importance > threshold]
                else:
                    method.fit(train_x, train_y)
                    selected_features = df_train.columns[method.support_].tolist()

                print(f"Selected features ({method_name}):", selected_features)
                if len(selected_features) > len(best_selected_features):
                    best_selected_features = selected_features
                    # Prepare data for model fitting
                    
                # print("Original features:", df_train.columns.tolist())
                
                # Ensure self.y_ is updated based on transformed dataset
    
                # X_train, X_test, y_train, y_test = train_test_split(self.X_, self.y_, test_size=0.2, random_state=42)
                # Train and evaluate using TabPFNClassifier
                # train_x, train_y = data.get_X_y(df_train, target_name)
                
                classifier = TabPFNClassifier(device="cpu")
                train_x_for_classifier = train_x[selected_features]
                test_x_for_classifier = test_x[selected_features]
                classifier.fit(train_x_for_classifier, train_y)
                y_pred = classifier.predict(test_x_for_classifier)
                accuracy = accuracy_score(test_y, y_pred)
                print(f"TabPFNClassifier Accuracy ({method_name}):", accuracy)

                if accuracy > best_score:
                    best_score = accuracy
                    best_selected_features = selected_features

            self.feature_names = best_selected_features
            print("Best selected features:", self.feature_names)

            train_x = train_x[self.feature_names]
            X = train_x.values
            y = y.astype(int)

            self.classes_ = unique_labels(y)
            self.base_classifier.fit(X, y)

            return self
# causility and then feature selection:
    def fit_causility_feature_selection(self, X, y, df_test,  model_attributes_file,df_train_file,causal_method, dataset_description, feature_names, target_name, disable_caafe=False):
        """
        Fit the model to the training data and save intermediate processed data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.
        dataset_description : str
            A description of the dataset.
        feature_names : List[str]
            The names of the features in the dataset.
        target_name : str
            The name of the target variable in the dataset.
        disable_caafe : bool, optional
            Whether to disable the CAAFE algorithm, by default False.
        """
        self.dataset_description = dataset_description
        self.feature_names = list(feature_names)
        self.target_name = target_name
        

        self.X_ = X
        self.y_ = y

        # Define file paths for caching processed data and attributes
        processed_file = df_train_file
        attributes_file = model_attributes_file

        # Load cached data if available
        if os.path.exists(processed_file) and os.path.exists(attributes_file):
            print("Loading pre-processed dataset and attributes from files...")
            df_train = pd.read_parquet(processed_file)
            with open(attributes_file, 'rb') as f:
                self.code, self.mappings = pickle.load(f)
            df_train = self.remove_duplicate_columns(df_train)
            df_train, _ = remove_nan_values(df_train, None)
            self.feature_names = [col for col in df_train.columns if col != target_name]
            print(f"Generated Features (All): {self.feature_names}")

        else:
            print("Processing dataset and saving for future use...")

            # Create initial dataframe
            df_train = pd.DataFrame(X, columns=self.feature_names)
            df_train[target_name] = y

            if disable_caafe:
                self.code = ""
            else:
                self.code, prompt, messages = generate_features(
                    ["dataset", X, y, [], self.feature_names + [target_name], {}, dataset_description],
                    df_train,
                    model=self.llm_model,
                    iterative=self.iterations,
                    metric_used=auc_metric,
                    iterative_method=self.base_classifier,
                    display_method="markdown",
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                )

            df_train = run_llm_code(self.code, df_train)

            # Convert dataset to numeric format
            df_train, _, self.mappings = make_datasets_numeric(
                df_train, df_test=None, target_column=target_name, return_mappings=True
            )

            # Save processed data and attributes
            df_train.to_parquet(processed_file, index=False)
            with open(attributes_file, 'wb') as f:
                pickle.dump((self.code, self.mappings), f)
            print("Processed dataset and attributes saved.")
                # Store feature names after new code execution
                
        print("Performing causal analysis...")
        selected_causal_features = causal_feature_selection(df_train, target_name, self.feature_names, causal_method)
        # print("All Features:", self.feature_names)
        print("Causal-Selected Features:", selected_causal_features)
        df_train = df_train[selected_causal_features + [target_name]]
        only_in_list1 = set(self.feature_names) - set(selected_causal_features)
        # print(f"The features different between original features and causality features are: {only_in_list1}") 
        # Feature selection experiments
        feature_selection_methods = {
            "RFE": RFE(RandomForestClassifier(), n_features_to_select=10),
            "LassoCV": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0)),
            "ElasticNetCV": make_pipeline(StandardScaler(), ElasticNetCV(cv=5, random_state=0)),
            "Mutual Information": mutual_info_classif,
            "Boruta": BorutaPy(RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5), n_estimators='auto', random_state=0),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=0),
            "Decision Tree": DecisionTreeClassifier(random_state=0)
        }
        train_x, train_y = split_target_column(df_train, target_name)
        test_x, test_y = split_target_column(df_test, target_name)

        # train_x, train_y = data.get_X_y(df_train, target_name)
        # test_x, test_y  = data.get_X_y(df_test, target_name)

        test_x = run_llm_code(self.code, test_x)
        test_x, _, _ = make_datasets_numeric(
            test_x, df_test=None, target_column=target_name, return_mappings=True
        )
        classifier = TabPFNClassifier(device="cpu")
        train_x = train_x[selected_causal_features]
        test_x = test_x[selected_causal_features]
        train_x, test_x = remove_nan_values(train_x, test_x)
        classifier.fit(train_x, train_y)
        y_pred = classifier.predict(test_x)
        accuracy = accuracy_score(test_y, y_pred)
        print(f"TabPFNClassifier Accuracy (causality):", accuracy)
        self.feature_names = [col for col in train_x.columns if col != target_name]

        best_selected_features = []
        best_score = -float("inf")
        for method_name, method in feature_selection_methods.items():
            print("")
            print(f"Applying feature selection using {method_name}...")
            if method_name in ["RFE"]:
                method.fit(train_x, train_y)
                        # Get the selected features
                selected_features = train_x.columns[method.support_].tolist()
            elif method_name in ["LassoCV"]:
                method.fit(train_x, train_y)
                        # Get the selected features
                selected_features = [
                    feature
                    for feature, coef in zip(self.feature_names, method.named_steps['lassocv'].coef_)
                    if coef != 0
                    ]
            elif method_name in ["ElasticNetCV"]:
                method.fit(train_x, train_y)
                selected_features = [feature for feature, coef in zip(self.feature_names, method.named_steps['elasticnetcv'].coef_) if coef != 0]
            elif method_name in ["Mutual Information"]:
                scores = method(train_x, train_y)
                threshold = np.median(scores)
                selected_features = [feature for feature, score in zip(self.feature_names, scores) if score > threshold]
            elif method_name in ["Boruta"]:
                method.fit(train_x, train_y)
                selected_features = [feature for feature, support in zip(self.feature_names, method.support_) if support]
            elif method_name in ["Gradient Boosting", "Decision Tree"]:
                method.fit(train_x, train_y)
                importances = method.feature_importances_
                threshold = np.mean(importances)
                selected_features = [feature for feature, importance in zip(self.feature_names, importances) if importance > threshold]
            else:
                method.fit(train_x, train_y)
                selected_features = df_train.columns[method.support_].tolist()

            print(f"Selected features ({method_name}):", selected_features)
            only_in_list1 = set(selected_causal_features) - set(selected_features)
            # print(f"The method is: {method_name}, selected features dif: {only_in_list1}") 

            if len(selected_features) > len(best_selected_features):
                best_selected_features = selected_features
                # Prepare data for model fitting
                
            # print("Original features:", df_train.columns.tolist())
            
             # Ensure self.y_ is updated based on transformed dataset
 
            # X_train, X_test, y_train, y_test = train_test_split(self.X_, self.y_, test_size=0.2, random_state=42)
            # Train and evaluate using TabPFNClassifier
            # train_x, train_y = data.get_X_y(df_train, target_name)
            
            classifier = TabPFNClassifier(device="cpu")
            train_x_for_classifier = train_x[selected_features]
            test_x_for_classifier = test_x[selected_features]
            classifier.fit(train_x_for_classifier, train_y)
            y_pred = classifier.predict(test_x_for_classifier)
            accuracy = accuracy_score(test_y, y_pred)
            print(f"TabPFNClassifier Accuracy ({method_name}):", accuracy)

            if accuracy > best_score:
                best_score = accuracy
                best_selected_features = selected_features

        self.feature_names = best_selected_features
        print("Best selected features:", self.feature_names)

        train_x = train_x[self.feature_names]
        X = train_x.values
        y = y.astype(int)

        self.classes_ = unique_labels(y)
        self.base_classifier.fit(X, y)

        return self

#feature selection and then causality:
    def fit_feature_selection_causality(self, X, y, df_test, model_attributes_file,df_train_file, causal_method, dataset_description, feature_names, target_name, disable_caafe=False):
        """
        Fit the model to the training data and save intermediate processed data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.
        dataset_description : str
            A description of the dataset.
        feature_names : List[str]
            The names of the features in the dataset.
        target_name : str
            The name of the target variable in the dataset.
        disable_caafe : bool, optional
            Whether to disable the CAAFE algorithm, by default False.
        """
        self.dataset_description = dataset_description
        self.feature_names = list(feature_names)
        self.target_name = target_name
        

        self.X_ = X
        self.y_ = y

        # Define file paths for caching processed data and attributes
        processed_file = df_train_file
        attributes_file = model_attributes_file

        # Load cached data if available
        if os.path.exists(processed_file) and os.path.exists(attributes_file):
            print("Loading pre-processed dataset and attributes from files...")
            df_train = pd.read_parquet(processed_file)
            with open(attributes_file, 'rb') as f:
                self.code, self.mappings = pickle.load(f)
            df_train = self.remove_duplicate_columns(df_train)

            self.feature_names = [col for col in df_train.columns if col != target_name]
            print(f"Generated Features (All): {self.feature_names}")

        else:
            print("Processing dataset and saving for future use...")

            # Create initial dataframe
            df_train = pd.DataFrame(X, columns=self.feature_names)
            df_train[target_name] = y

            if disable_caafe:
                self.code = ""
            else:
                self.code, prompt, messages = generate_features(
                    ["dataset", X, y, [], self.feature_names + [target_name], {}, dataset_description],
                    df_train,
                    model=self.llm_model,
                    iterative=self.iterations,
                    metric_used=auc_metric,
                    iterative_method=self.base_classifier,
                    display_method="markdown",
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                )

            df_train = run_llm_code(self.code, df_train)

            # Convert dataset to numeric format
            df_train, _, self.mappings = make_datasets_numeric(
                df_train, df_test=None, target_column=target_name, return_mappings=True
            )

            # Save processed data and attributes
            df_train.to_parquet(processed_file, index=False)
            with open(attributes_file, 'wb') as f:
                pickle.dump((self.code, self.mappings), f)
            print("Processed dataset and attributes saved.")
                # Store feature names after new code execution
                
        # Feature selection experiments
        feature_selection_methods = {
            "RFE": RFE(RandomForestClassifier(), n_features_to_select=10),
            "LassoCV": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0)),
            "ElasticNetCV": make_pipeline(StandardScaler(), ElasticNetCV(cv=5, random_state=0)),
            "Mutual Information": mutual_info_classif,
            "Boruta": BorutaPy(RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5), n_estimators='auto', random_state=0),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=0),
            "Decision Tree": DecisionTreeClassifier(random_state=0)
        }
        train_x, train_y = split_target_column(df_train, target_name)
        test_x, test_y = split_target_column(df_test, target_name)

        # train_x, train_y = data.get_X_y(df_train, target_name)
        # test_x, test_y  = data.get_X_y(df_test, target_name)

        test_x = run_llm_code(self.code, test_x)
        test_x, _, _ = make_datasets_numeric(
            test_x, df_test=None, target_column=target_name, return_mappings=True
        )
        train_x, test_x = remove_nan_values(train_x, test_x)

        best_selected_features = []
        best_score = -float("inf")
        for method_name, method in feature_selection_methods.items():
            print("")
            print(f"Applying feature selection using {method_name}...")
            if method_name in ["RFE"]:
                method.fit(train_x, train_y)
                        # Get the selected features
                selected_features = train_x.columns[method.support_].tolist()
            elif method_name in ["LassoCV"]:
                method.fit(train_x, train_y)
                        # Get the selected features
                selected_features = [
                    feature
                    for feature, coef in zip(self.feature_names, method.named_steps['lassocv'].coef_)
                    if coef != 0
                    ]
            elif method_name in ["ElasticNetCV"]:
                method.fit(train_x, train_y)
                selected_features = [feature for feature, coef in zip(self.feature_names, method.named_steps['elasticnetcv'].coef_) if coef != 0]
            elif method_name in ["Mutual Information"]:
                scores = method(train_x, train_y)
                threshold = np.median(scores)
                selected_features = [feature for feature, score in zip(self.feature_names, scores) if score > threshold]
            elif method_name in ["Boruta"]:
                method.fit(train_x, train_y)
                selected_features = [feature for feature, support in zip(self.feature_names, method.support_) if support]
            elif method_name in ["Gradient Boosting", "Decision Tree"]:
                method.fit(train_x, train_y)
                importances = method.feature_importances_
                threshold = np.mean(importances)
                selected_features = [feature for feature, importance in zip(self.feature_names, importances) if importance > threshold]
            else:
                method.fit(train_x, train_y)
                selected_features = df_train.columns[method.support_].tolist()

            
            if len(selected_features) > len(best_selected_features):
                best_selected_features = selected_features
                # Prepare data for model fitting
                
            # print("Original features:", df_train.columns.tolist())
            
             # Ensure self.y_ is updated based on transformed dataset
 
            # X_train, X_test, y_train, y_test = train_test_split(self.X_, self.y_, test_size=0.2, random_state=42)
            # Train and evaluate using TabPFNClassifier
            # train_x, train_y = data.get_X_y(df_train, target_name)
            train_x_for_classifier = train_x[selected_features]
            test_x_for_classifier = test_x[selected_features]
            df_train ,_  = remove_nan_values(df_train, None)
            df_train_causalModel = df_train[selected_features + [target_name]]

            selected_causal_features = causal_feature_selection(df_train_causalModel,target_name, selected_features, causal_method)
            only_in_list1 = set(selected_features) - set(selected_causal_features)
            # print(f"The method is: {method_name}, the different features are: {only_in_list1}") 
            if len(selected_causal_features) == 0:
                # print("Weak or No Causal Relationship ({method_name})")
                continue
            print(f"Selected features ({method_name}):", selected_causal_features)
            train_x_for_classifier = train_x_for_classifier[selected_causal_features]
            test_x_for_classifier = test_x_for_classifier[selected_causal_features]

            classifier = TabPFNClassifier(device="cpu")
            classifier.fit(train_x_for_classifier, train_y)
            y_pred = classifier.predict(test_x_for_classifier)
            accuracy = accuracy_score(test_y, y_pred)
            print(f"TabPFNClassifier Accuracy ({method_name}):", accuracy)

            if accuracy > best_score:
                best_score = accuracy
                best_selected_features = selected_features

        self.feature_names = best_selected_features
        print("Best selected features:", self.feature_names)

        train_x = train_x[self.feature_names]
        X = train_x.values
        y = y.astype(int)

        self.classes_ = unique_labels(y)
        self.base_classifier.fit(X, y)

        return self


    def predict_preprocess(self, X):
        """
        Helper functions for preprocessing the data before making predictions.

        Parameters:
        X (pandas.DataFrame): The DataFrame to make predictions on.

        Returns:
        numpy.ndarray: The preprocessed input data.
        """
        # check_is_fitted(self)

        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=self.X_.columns)
        X, _ = split_target_column(X, self.target_name)

        X = run_llm_code(
            self.code,
            X,
        )
        X = X[self.feature_names]
        X = make_dataset_numeric(X, mappings=self.mappings)

        X = X.values

        # Input validation
        # X = check_array(X)
        
        return X

    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict_proba(X)

    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict(X)
def causal_feature_selection(df_train, target_name, feature_names, method='individual'):
    """
    Select causal features using different methods: individual estimation, DAG, PSM, or IV.
    
    Parameters:
    df_train (pd.DataFrame): The dataset.
    target_name (str): The target variable.
    feature_names (list): List of feature names.
    method (str): The causality method to use ('individual', 'dag', 'psm', 'iv').
    
    Returns:
    list: Selected causal features.
    """
    from dowhy import CausalModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import statsmodels.api as sm
    from statsmodels.sandbox.regression.gmm import IV2SLS
    
    selected_causal_features = []
    
    if method == 'individual':
        for feature in feature_names:
            model = CausalModel(
                data=df_train,
                treatment=feature,  # Estimate causality for each feature separately
                outcome=target_name
            )
            identified_estimand = model.identify_effect()
            causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
            if abs(causal_estimate.value) > 0.1:
                # print(f"Causal Estimate for {feature}: {causal_estimate.value}")
                selected_causal_features.append(feature)
    
    elif method == 'dag':
        causal_graph = """
        digraph {
            Wifes_education -> Contraceptive_method_used;
            Husbands_education -> Contraceptive_method_used;
            Wifes_religion -> Contraceptive_method_used;
            Husbands_occupation -> Contraceptive_method_used;
            Standard_of_living_index -> Contraceptive_method_used;
        }
        """
        model = CausalModel(
            data=df_train,
            graph=causal_graph,
            treatment=feature_names,
            outcome=target_name
        )
        model.view_model()
        identified_estimand = model.identify_effect()
        causal_estimates = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        selected_causal_features = [
            feature for feature, estimate in zip(feature_names, causal_estimates.value)
            if abs(estimate) > 0.1
        ]
    elif method == 'psm':
        for feature in feature_names:
            log_reg = LogisticRegression(multi_class="ovr")  # Handle multi-class case
            log_reg.fit(df_train[[feature]], df_train[target_name])
            
            # Ensure target is binarized for AUC computation
            y_proba = log_reg.predict_proba(df_train[[feature]])

            if y_proba.shape[1] > 2:  # Multi-class scenario
                score = roc_auc_score(df_train[target_name], y_proba, multi_class="ovr")
            else:  # Binary classification
                score = roc_auc_score(df_train[target_name], y_proba[:, 1])

            if score > 0.6:  # Threshold for strong causal effect
                selected_causal_features.append(feature)

            
    return selected_causal_features

def remove_nan_values(train_x, test_x=None):
    """
    Removes NaN values from train_x and test_x using mean imputation.

    Parameters:
    train_x (pd.DataFrame): Training feature set
    test_x (pd.DataFrame): Test feature set

    Returns:
    pd.DataFrame, pd.DataFrame: train_x and test_x without NaN values
    """
    from sklearn.impute import SimpleImputer
    
    # Initialize imputer (using mean, but can use median or most_frequent)
    imputer = SimpleImputer(strategy="mean")

    # Fit imputer on train_x and transform both train_x and test_x
    if  train_x is not None:
        train_x = pd.DataFrame(imputer.fit_transform(train_x), columns=train_x.columns)
    if test_x is not None:
        test_x = pd.DataFrame(imputer.transform(test_x), columns=test_x.columns)

    return train_x, test_x

