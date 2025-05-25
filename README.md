# AdaptiveGS

This program implements a Stacking ensemble learning method for model selection based on the PR index (PCC/NRMSE) for genomic selection. Genomic selection (GS) is a crucial technique in modern molecular breeding, and stacking ensemble learning (SEL) has proven to be a powerful GS approach by integrating multiple base learners (BLs) to capture the complex relationships between genotypes and phenotypes. However, selecting the optimal BLs remains a challenge due to the lack of an effective and unified framework.

**AdaptiveGS** is an adaptive and explainable data-driven BL selection strategy designed to enhance GS accuracy. It employs the **PR index**, which combines the **Pearson correlation coefficient (PCC)** and **root mean square error (RMSE)**, to pre-screen and select the top-performing BLs for the stacking GS framework. This strategy ensures that the most suitable BLs are chosen for different species and traits, improving overall prediction performance.

**Citation** Zhen Yang, Mei Song, Xianggeng Huang, Quanrui Rao, Shanghui Zhang, Zhongzheng Zhang, Chenyang Wang, Wenjia Li, Ran Qin, Chunhua Zhao, Yongzhen Wu, Han Sun, Guangchen Liu, Fa Cui, AdaptiveGS: an explainable genomic selection framework based on adaptive stacking ensemble machine learning, 2025,in submiting.

**Suggestions and Problem Feedback** liuguangchen@ldu.edu.cn

---

## Key Features

- **Adaptive Base Learner Selection**: Automatically screens and selects the top N (default: 3) out of 7 candidate algorithms based on the PR index, ensuring optimal base learners for specific datasets.
- **Improved Prediction Accuracy**: Achieved an average prediction accuracy of 0.704 across ten traits, with a notable improvement of 14.4% compared to traditional GS models.
- **Explainable Model Interpretation**: Utilizes **SHapley Additive exPlanations (SHAP)** and **Permutation Importance (PI)** to identify key SNPs associated with target traits, providing insights into model decisions.
- **Open-Source and User-Friendly**: Available as a Python package, designed for easy integration into genomic selection pipelines.

---

## Base Functions

AdaptiveGS is organized into several modular Python scripts:

- `adaptiveGS.py`: Implements the integrated Stacking-based machine learning framework for model performance evaluation and hyperparameter optimization. This is the main script for the AdaptiveGS stacking methodology.
- `adaptiveGS_dl.py`: Provides genomic prediction models based on Deep Learning architectures (CNNs, MLPs, Transformers), offering a contrasting approach to the Stacking ensemble model.
- `adaptiveGS_linear_models.py`: Implements R-based genomic prediction models (e.g., BayesB, GBLUP, RRBLUP) using `rpy2` to integrate and execute R models directly from Python.
- `SHAP_analysis.py`: Performs SHAP-based feature importance and model interpretation for pre-trained genomic prediction models. It generates various visualizations (bar plots, beeswarm plots, heatmaps, decision plots, violin plots) to explain feature contributions and model decisions. This script now supports multiple SHAP explainer types (`kernel`, `permutation`).
- `PI_analysis.py`: Conducts permutation importance analysis by shuffling feature values to measure their impact on model performance. It loads pre-trained models and generates complementary visualizations (box plots, violin plots) to assess feature contributions from a different perspective.

---

## Getting Started

### 1. Data Preparation

Ensure your genotype and phenotype data are prepared as `.csv` files.

- **Genotype Data (e.g., `gy_geno.csv`):**
  - The first column must contain unique sample/line IDs (e.g., `ID_line`).
  - Subsequent columns should represent genetic markers (SNPs) as features.
- **Phenotype Data (e.g., `gy_pheno.csv`):**
  - The first column must also contain the same unique sample/line IDs, matching those in the genotype file.
  - One or more columns should contain the target phenotype values (e.g., `yield`).
- **Example Data Source:** The project uses data from Lozada DN, Ward BP, Carter AH (2020) Gains through selection for grain yield in a winter wheat breeding program. PLoS ONE 15:e0221603. https://doi.org/1371/journal.pone.0221603.

### 2. Environment Setup

All necessary Python libraries can be installed via pip:

```bash
pip install scikit-learn pandas numpy lightgbm xgboost shap matplotlib seaborn rpy2 memory_profiler # rpy2 is only needed for adaptiveGS_linear_models.py
```

* **For `adaptiveGS_linear_models.py` (R-based models):**
  
  * You must have **R installed** on your system. Our experiments were conducted with `R-4.2.3`. It's recommended to use this version or newer for compatibility.`
  
  * Set the `R_HOME` environment variable to your R installation path.
  
  * Ensure the R packages used by the models (`BWGS`, `rrBLUP`) are installed within your R environment, a brief installation introduction is as follows:
    
    - **`BWGS`**:
      
      - **Installation:**
        
        ```R
        install.packages("BWGS")
        ```
      
      - **Source:** [BWGS GitHub Repository](https://github.com/vangiangtran/BWGS?tab=readme-ov-file)
      
      - **Primary Citation :**
        
        - Charmet G, Tran LG, Auzanneau J, Rincent R, Bouchet S. BWGS: A R package for genomic selection and its application to a wheat breeding programme. PLoS One. 2020 Apr 23;15(4):e0232422. doi: 10.1371/journal.pone.0232422. PMID: 32240182; PMCID: PMC7141418.*
    
    - **`rrBLUP`**:
      
      
      - **Installation:**
        
        ```R
        install.packages("rrBLUP")
        ```
      
      - **Source:** [rrBLUP CRAN Page](https://cran.r-project.org/web/packages/rrBLUP/index.html)
      
      - **Primary Citation:**
        
        - Endelman, J. B. (2011). *Ridge Regression and Other Kernels for Genomic Selection with R Package rrBLUP. The Plant Genome Journal, 4(3), 250.* doi:10.3835/plantgenome2011.08.0024](https://doi.org/10.3835/plantgenome2011.08.0024)

### 3. Directory Structure

The scripts expect a specific directory structure for data and model checkpoints:

```
.
├── datasets/
│   └── Wheat/
│       ├── gy_geno.csv
│       └── gy_pheno.csv
├── checkpoints/
│   └── {dataset_id}_{target_col}/
│       └── {model_type}/
│           └── {model_type}.joblib or {model_type}.pth  # Trained model files
├── results/                     # Output directory for performance metrics and plots
│   └── {dataset_id}_{target_col}/
│       └── {model_type}/
│           ├── shap_analysis_auto/
│           ├── shap_analysis_kernel/
│           └── shap_analysis_permutation/
├── R_utils/                     # Contains R scripts for linear models (e.g., BayesB.R)
├── adaptiveGS.py
├── adaptiveGS_dl.py
├── adaptiveGS_linear_models.py
├── SHAP_analysis.py
└── PI_analysis.py
```

---

## How to Use

### adaptiveGS.py (Adaptive Stacking Ensemble)

This script performs the core AdaptiveGS stacking.

1. **Dataset Configuration (`dataset_paths`, `dataset_config`):**
   Update the file paths to your genotype and phenotype CSV files and define the target phenotype column.
   
   ```python
   dataset_paths = {
       "gy_geno": "datasets/Wheat/gy_geno.csv", # Adjust path to your geno data
       "gy_pheno": "datasets/Wheat/gy_pheno.csv" # Adjust path to your pheno data
   }
   dataset_config = [
       {"id": "gy", "target": "yield"} # Adjust dataset ID and target phenotype
   ]
   ```

2. **Output Path (`save_path`):**
   Specify the base directory where results and trained models will be saved.
   
   ```python
   save_path = 'results_directory' # e.g., 'results'
   ```

3. **Hyperparameter Tuning & Stacking Settings:**
   Adjust parameters for Optuna hyperparameter optimization and the stacking ensemble itself.
   
   ```python
   n_trials_optuna = 20        # Number of Optuna trials per base learner
   cv_folds_optuna = 10        # Cross-validation folds for Optuna tuning
   base_select_model_n = 3     # Number of top-performing base learners to select for stacking
   cv_folds_stacking = 10      # Cross-validation folds for the stacking ensemble
   ```

4. **Model & Parameter Customization (`get_model_configs()`):**
   Add, remove, or refine the candidate base learners and their hyperparameter search spaces.
   
   ```python
   def get_model_configs():
       return {
           'NewModel': {
               'model': NewEstimator(), # Your new scikit-learn compatible model
               'param_distributions': {
                   'param_name': optuna.distributions.FloatDistribution(0.1, 10.0)
               }
           },
           # ... existing models (LightGBM, XGBoost, CatBoost, RandomForest, etc.)
       }
   ```

5. **Run:**
   
   ```bash
   python adaptiveGS.py
   ```

6. **Results Viewing:**
   After execution, an Excel file named `ML_performance.xlsx` will be generated in the specified `save_path`, containing detailed evaluation results for each base learner and the final Stacking model. Trained models will be saved in the `checkpoints` directory.

### adaptiveGS_dl.py (Deep Learning Models)

This script trains and evaluates deep learning models for genomic prediction.

1. **Dataset Configuration (`dataset_paths`, `dataset_config`):**
   (Same as `adaptiveGS.py`)

2. **Output Path (`save_path`):**
   (Same as `adaptiveGS.py`)

3. **Optimization & Training Parameters:**
   Adjust the intensity of hyperparameter search and model training parameters like epochs and batch size.
   
   ```python
   n_splits = 10  # Cross-validation folds for Optuna tuning
   n_trials = 20  # Number of Optuna trials per deep learning model
   # Initial/fallback epochs/batch_size (can be overridden by Optuna tuning)
   ORIGINAL_HYPERPARAMS = {
       'CNN': {'num_epochs': 150, 'batch_size': 64, ...},
       # ... other DL models
   }
   ```

4. **Model Architectures & Hyperparameter Ranges (`models_to_train`):**
   Modify or add deep learning models (CNN, MLP, Transformer) and their corresponding hyperparameter distributions for Optuna.
   
   ```python
   models_to_train = {
       'CNN': (CNNModel, {
           'num_conv_layers': optuna.distributions.IntDistribution(2, 4),
           'dropout': optuna.distributions.FloatDistribution(0.1, 0.7),
           'lr': optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
           'batch_size': optuna.distributions.IntDistribution(64, 128, log=True),
           'initial_out_channels': optuna.distributions.CategoricalDistribution([16, 32, 64]),
       }),
       # Add your custom models or modify existing ones...
   }
   ```

5. **Run:**
   
   ```bash
   python adaptiveGS_dl.py
   ```

### adaptiveGS_linear_models.py (R-based Linear Models)

This script evaluates R-based genomic prediction models using `rpy2`.

1. **R Environment Setup:**
   **Crucially**, set the `R_HOME` environment variable to your R installation path before running the script. This allows `rpy2` to locate your R installation.
   
   ```python
   import os
   os.environ['R_HOME'] = r'D:\R\R-4.2.3' # Change this to your R installation directory
   # Also ensure necessary R packages are installed, e.g., install.packages("BGLR")
   ```

2. **Dataset Configuration (`dataset_paths`, `dataset_config`):**
   (Same as `adaptiveGS.py`)

3. **Output Path (`checkpoints_save_path`):**
   Define the base directory for saving model outputs and performance metrics for R models.
   
   ```python
   checkpoints_save_path = 'your_results_directory' # e.g., 'results_R_models'
   ```

4. **Models to Evaluate (`models_to_evaluate`):**
   Choose which R models (e.g., `BayesB`, `GBLUP`, `RRBLUP`) to run.
   
   ```python
   models_to_evaluate = ['BayesB', 'GBLUP', 'RRBLUP'] # Add/remove R model names
   ```

5. **R Script Files:**
   Ensure corresponding R script files (e.g., `BayesB.R`, `GBLUP.R`) are present in the `./R_utils/` directory.

6. **Run:**
   
   ```bash
   python adaptiveGS_linear_models.py
   ```

### shap_analysis.py and permutation_analysis.py (Model Interpretability)

These scripts provide explainable AI capabilities (SHAP and Permutation Importance) for **pre-trained models**.

**Prerequisite:** Ensure your trained models (from `adaptiveGS.py`, `adaptiveGS_dl.py`, or `adaptiveGS_linear_models.py`) are located in the `checkpoints` directory, following the `checkpoints/{dataset_id}_{target_col}/{model_type}/{model_type}.joblib` structure.

1. **Dataset Configuration (`dataset_paths`, `dataset_config`):**
   Update `dataset_paths` with your genotype and phenotype CSV file locations. Then, define which specific dataset IDs and their target phenotype columns you want to analyze in `dataset_config`.
   
   ```python
   dataset_paths = {
       "gy_geno": "datasets/Wheat/gy_geno.csv",
       "gy_pheno": "datasets/Wheat/gy_pheno.csv"
   }
   dataset_config = [
       {"id": "gy", "target": "yield"}
   ]
   ```

2. **Models to Analyze (`model_types`):**
   List the names of the trained models (e.g., `'Stacking'`, `'LightGBM'`, `'CNN'`) for which you want to generate interpretability plots. These names must match the folder names under `checkpoints/` where your models were saved.
   
   ```python
   model_types = [
       "Stacking",
       "LightGBM", # Add other model names as needed
   ]
   ```

3. **Run `shap_analysis.py`:**
   This script will generate SHAP plots for all specified models. The updated script automatically iterates through `explainer_type` (`kernel`, `permutation`), saving results in separate subdirectories (e.g., `shap_analysis_kernel`).
   
   ```bash
   python shap_analysis.py
   ```

4. **Run `permutation_analysis.py`:**
   This script will generate Permutation Importance plots for all specified models.
   
   ```bash
   python permutation_analysis.py
   ```
