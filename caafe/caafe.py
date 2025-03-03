import copy
import numpy as np

from sklearn.model_selection import RepeatedKFold
from .caafe_evaluate import (
    evaluate_dataset,get_leave_one_out_importance
)
from .run_llm_code import run_llm_code
from transformers import pipeline
import requests
from google import genai
from dowhy import CausalModel
import re
import os
from dotenv import load_dotenv
def get_prompt(
    df, ds, iterative=1, data_description_unparsed=None, samples=None, **kwargs
):
    how_many = (
        "up to 10 useful columns. Generate as many features as useful for downstream classifier, but as few as necessary to reach good performance."
        if iterative == 1
        else "exactly one useful column"
    )
    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}
    
This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {int(len(df))}
    
This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{ds[4][-1]}\".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore.

Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to classify \"{ds[4][-1]}\" according to dataset description and attributes.)
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)
(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
```end

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```end

Each codeblock generates {how_many} and can drop unused columns (Feature selection).
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""


# Each codeblock either generates {how_many} or drops bad columns (Feature selection).


def build_prompt_from_df(ds, df, iterative=1):
    data_description_unparsed = ds[-1]
    feature_importance = {}  # xgb_eval(_obj)

    samples = ""
    df_ = df.head(10)
    for i in list(df_):
        # show the list of values
        nan_freq = "%s" % float("%.2g" % (df[i].isna().mean() * 100))
        s = df_[i].tolist()
        if str(df[i].dtype) == "float64":
            s = [round(sample, 2) for sample in s]
        samples += (
            f"{df_[i].name} ({df[i].dtype}): NaN-freq [{nan_freq}%], Samples {s}\n"
        )

    kwargs = {
        "data_description_unparsed": data_description_unparsed,
        "samples": samples,
        "feature_importance": {
            k: "%s" % float("%.2g" % feature_importance[k]) for k in feature_importance
        },
    }

    prompt = get_prompt(
        df,
        ds,
        data_description_unparsed=data_description_unparsed,
        iterative=iterative,
        samples=samples,
    )

    return prompt


def generate_features(
    ds,
    df,
    model="gpt-3.5-turbo",
    just_print_prompt=False,
    iterative=1,
    metric_used=None,
    iterative_method="logistic",
    display_method="markdown",
    n_splits=10,
    n_repeats=2,
):
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print

    assert (
        iterative == 1 or metric_used is not None
    ), "metric_used must be set if iterative"

    prompt = build_prompt_from_df(ds, df, iterative=iterative)

    if just_print_prompt:
        code, prompt = None, prompt
        return code, prompt, None


    def generate_code(messages, model="default-model"):
        if model == "skip":
            return ""
        

        load_dotenv()
        api_key = os.getenv("API_KEY")
        client = genai.Client(api_key)
        
     
        try:
            response = client.models.generate_content(
            model="gemini-2.0-flash", contents=messages
        )
        except Exception as e: 
            print(e)
        if response.text:
            # completion = response.json()['choices'][0]['text']
            code = response.text.replace("```python", "").replace("```", "").replace("end", "")
            return code
        else:
            print("Failed to retrieve completion:", response.text)
            return ""


    def execute_and_evaluate_code_block(full_code, code):
        old_accs, old_rocs, accs, rocs = [], [], [], []
        ss = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        for (train_idx, valid_idx) in ss.split(df):
            df_train, df_valid = df.iloc[train_idx], df.iloc[valid_idx]

            # Remove target column from df_train
            target_train = df_train[ds[4][-1]]
            target_valid = df_valid[ds[4][-1]]
            df_train = df_train.drop(columns=[ds[4][-1]])
            df_valid = df_valid.drop(columns=[ds[4][-1]])

            df_train_extended = copy.deepcopy(df_train)
            df_valid_extended = copy.deepcopy(df_valid)

            try:
                df_train = run_llm_code(
                    full_code,
                    df_train,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )
                df_valid = run_llm_code(
                    full_code,
                    df_valid,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )
                df_train_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_train_extended,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )
                df_valid_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_valid_extended,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )
                # Store feature names after new code execution
                def standardize_column_name(col_name):
                    """Converts column names to lowercase and removes special characters like underscores and dashes."""
                    return re.sub(r'[^a-zA-Z0-9]', '', col_name.lower())

                def remove_duplicate_columns(df):
                    """Removes duplicate columns that differ only by capitalization or special characters."""
                    standardized_cols = {}
                    cols_to_drop = []

                    for col in df.columns:
                        std_col = standardize_column_name(col)
                        if std_col in standardized_cols:
                            cols_to_drop.append(col)  # Mark this column for removal
                        else:
                            standardized_cols[std_col] = col  # Store the first occurrence

                    return df.drop(columns=cols_to_drop)

                # Apply to both DataFrames
                df_train_extended = remove_duplicate_columns(df_train_extended)
                df_valid_extended = remove_duplicate_columns(df_valid_extended)

            except Exception as e:
                display_method(f"Error in code execution. {type(e)} {e}")
                display_method(f"```python\n{format_for_display(code)}\n```\n")
                return e, None, None, None, None

            # Add target column back to df_train
            df_train[ds[4][-1]] = target_train
            df_valid[ds[4][-1]] = target_valid
            df_train_extended[ds[4][-1]] = target_train
            df_valid_extended[ds[4][-1]] = target_valid
            
            original_features = set(df_train.columns)
            new_features = set(df_train_extended.columns) - original_features
            from contextlib import contextmanager
            import sys, os

            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    result_old = evaluate_dataset(
                        df_train=df_train,
                        df_test=df_valid,
                        prompt_id="XX",
                        name=ds[0],
                        method=iterative_method,
                        metric_used=metric_used,
                        seed=0,
                        target_name=ds[4][-1],
                    )

                    result_extended = evaluate_dataset(
                        df_train=df_train_extended,
                        df_test=df_valid_extended,
                        prompt_id="XX",
                        name=ds[0],
                        method=iterative_method,
                        metric_used=metric_used,
                        seed=0,
                        target_name=ds[4][-1],
                    )
                finally:
                    sys.stdout = old_stdout
                    
            """ROC Improvement by using each feature: {importances}"""
            old_accs += [result_old["acc"]]
            old_rocs += [result_old["roc"]]
            accs += [result_extended["acc"]]
            rocs += [result_extended["roc"]]
        return None, rocs, accs, old_rocs, old_accs
    
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.",
    #     },
    #     {
    #         "role": "user",
    #         "content": prompt,
    #     },
    # ]
    messages ="You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible." +prompt
    display_method(f"*Dataset description:*\n {ds[-1]}")
    print(f"*Dataset description:*\n {ds[-1]}")
    n_iter = 5
    full_code = ""

    i = 0
    while i < n_iter:
        try:
            code = generate_code(messages)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            continue
        i = i + 1
        e, rocs, accs, old_rocs, old_accs = execute_and_evaluate_code_block(
            full_code, code
        )
        if e is not None:
            messages += """Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next feature (fixing error?):
                                ```python
                                """
                
            continue

        # importances = get_leave_one_out_importance(
        #    df_train_extended,
        #    df_valid_extended,
        #    ds,
        #    iterative_method,
        #    metric_used,
        # )
        # """ROC Improvement by using each feature: {importances}"""

        improvement_roc = np.nanmean(rocs) - np.nanmean(old_rocs)
        improvement_acc = np.nanmean(accs) - np.nanmean(old_accs)
        
        traget_column = ds[4][-1]
        
        # add_feature, causal_effect = causal_filtering(df_train_extended, new_features, traget_column)
        add_feature = True
        add_feature_sentence = "The code was executed and changes to ´df´ were kept."
        if improvement_roc + improvement_acc <= 0:
            add_feature = False
            add_feature_sentence = f"The last code changes to ´df´ were discarded. (Improvement: {improvement_roc + improvement_acc})"
           
        # display_method(
        #     "\n"
        #     + f"*Iteration {i}*\n"
        #     + f"```python\n{format_for_display(code)}\n```\n"
        #     + f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}, ACC {np.nanmean(old_accs):.3f}.\n"
        #     + f"Performance after adding features ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}.\n"
        #     + f"Improvement ROC {improvement_roc:.3f}, ACC {improvement_acc:.3f}.\n"
        #     + f"{add_feature_sentence}\n"
        #     + f"\n"
        # )
        print("\n"
            + f"*Iteration {i}*\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}, ACC {np.nanmean(old_accs):.3f}.\n"
            + f"Performance after adding features ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}.\n"
            + f"Improvement ROC {improvement_roc:.3f}, ACC {improvement_acc:.3f}.\n"
            + f"{add_feature_sentence}\n"
            + f"\n")

        if len(code) > 10:
            messages += """Performance after adding feature ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}. {add_feature_sentence}
            Next codeblock:
            """
        # if add_feature:
        full_code += code
        
        # if add_feature and (improvement_roc + improvement_acc > 0):
        #     full_code += code  # Keep feature
        # else:
        #     # df.drop(columns=[new_features], inplace=True)  # Remove non-causal feature


    return full_code, prompt, messages

def causal_filtering(df, new_features, target_column):
    filtered_features = []
    causal_effects = {}

    for feature in new_features:
        try:
            causal_model = CausalModel(
                data=df,
                treatment=feature,  # Process one feature at a time
                outcome=target_column,  # Target variable
                common_causes=list(df.columns.difference([feature, target_column])),
            )
            identified_estimand = causal_model.identify_effect()
            causal_estimate = causal_model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

            causal_effect = causal_estimate.value
            causal_effects[feature] = causal_effect  # Store causal effect value

            # Keep the feature if it has a significant causal effect
            if causal_effect > 0.01:  # Define a significance threshold
                filtered_features.append(feature)

        except Exception as e:
            print(f"Error in causal analysis for feature {feature}: {e}")
            causal_effects[feature] = None  # Mark as None if failed

    return len(filtered_features) > 0, causal_effects
