# Enhancing Feature Engineering with Large Language Models

## Overview
This project explores **feature engineering automation** using **Large Language Models (LLMs)** combined with **causal inference and feature selection techniques**.

We investigate whether LLMs, particularly **Gemini-2.0-flash**, can replace domain expertise in feature engineering by automatically generating and selecting meaningful features. Additionally, we examine the impact of **causal filtering** and **feature selection** on model performance.

## Datasets
This project evaluates four different datasets, each representing a distinct domain:

1. **Contraceptive Method Choice (CMC) [Index 2]**  
   - Demographic and socio-economic factors affecting contraceptive use.
2. **Diabetes [Index 4]**  
   - Patient medical records predicting diabetes diagnosis.
3. **Eucalyptus [Index 6]**  
   - Tree species characteristics and environmental factors.
4. **Airlines [Index 8]**  
   - Flight-related data predicting flight delays.

The **default dataset used in `main.ipynb` is CMC (Index 2)**, but this can be modified to analyze any of the other datasets by updating the dataset index in the `main.py` script.

## **Experiments Conducted**
We perform **three experiments** to analyze different approaches to feature selection and causal filtering:

1. **Feature Selection Only** – Apply various feature selection methods (e.g., RFE, LassoCV, Mutual Information) and evaluate accuracy.
2. **Causal Analysis First, Then Feature Selection** – Perform causality analysis before applying feature selection methods.
3. **Feature Selection First, Then Causal Analysis** – Standard feature selection methods are applied first, followed by causal filtering.

**Goal:** To determine whether integrating causality / feature selection improves model performance.

## **Results Notebooks**
The results of the **CMC dataset** are documented in **`main.ipynb`**, but **this notebook can be modified to analyze any dataset by changing the dataset index**.

Additionally, there are **four result notebooks**, each containing detailed plots and evaluations of the respective datasets:

- **`results_2.ipynb`** – CMC dataset (**Index 2**)
- **`results_4.ipynb`** – Diabetes dataset (**Index 4**)
- **`results_6.ipynb`** – Eucalyptus dataset (**Index 6**)
- **`results_8.ipynb`** – Airlines dataset (**Index 8**)

### **Modifying the Dataset in `main.ipynb`**
To run the experiments on a different dataset, **change the dataset index** in `main.py` as indicated in the comments:

| Dataset    | Index |
|------------|------|
| **CMC**        | 2 |
| **Diabetes**   | 4 |
| **Eucalyptus** | 6 |
| **Airlines**   | 8 |

Example (change `dataset_index` in `main.ipynb`):

```python
dataset_index = 4  # Change to 2, 4, 6, or 8
