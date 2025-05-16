# AdaptiveGS

This program implements a Stacking ensemble learning method for model selection based on the PR index (PCC/NRMSE)  for  genomic selection. Genomic selection (GS) is a crucial technique in modern molecular breeding, and stacking ensemble learning (SEL) has proven to be a powerful GS approach by integrating multiple base learners (BLs) to capture the complex relationships between genotypes and phenotypes. However, selecting the optimal BLs remains a challenge due to the lack of an effective and unified framework.

**AdaptiveGS** is an adaptive and explainable data-driven BL selection strategy designed to enhance GS accuracy. It employs the **PR index**, which combines the **Pearson correlation coefficient (PCC)** and **root mean square error (RMSE)**, to pre-screen and select the top-performing BLs for the stacking GS framework. This strategy ensures that the most suitable BLs are chosen for different species and traits, improving overall prediction performance.

**Citation**
Zhen Yang, Mei Song, Quanrui Rao, Shanghui Zhang, Zhongzheng Zhang, Chenyang Wang, Wenjia Li, Ran Qin, Chunhua Zhao, Yongzhen Wu, Han Sun, Guangchen Liu, Fa Cui, AdaptiveGS: an explainable genomic selection framework based on adaptive stacking ensemble machine learning, 2025,in submiting.

**Suggestions and Problem Feedback**
liuguangchen@ldu.edu.cn

## Key Features

- **Adaptive Base Learner Selection**: Automatically selects the top 3 out of 7 (or user-defined) candidate algorithms based on the PR index.
- **Improved Prediction Accuracy**: Achieved an average prediction accuracy of 0.811 across ten traits, with an improvement of 16.1% compared to other GS models.
- **Explainable Model Interpretation**: Utilizes **SHapley Additive exPlanations (SHAP)** to identify key SNPs associated with target traits.
- **Open-Source and User-Friendly**: Available as a Python package for easy integration into genomic selection pipelines.

## Basic Functions

adaptiveGS.py implements an integrated Stacking-based machine learning framework for model performance evaluation and hyperparameter optimization.

adaptiveGS_dl.py implements a genomic prediction model based on CNNs (Convolutional Neural Networks), MLPs (Multilayer Perceptual Machines), and Transformers, aiming to predict phenotype data from genotype data, in contrast to the Stacking Integrated Learning Model.

## How to use

**adaptiveGS.py**

Data preparation: store the gene data (example format: gy_geno.xlsx) and phenotype data (example format: gy_pheno.xlsx) under the appropriate paths and make sure that the data format and column names are correct in the Excel file.
（Data source: Lozada DN, Ward BP, Carter AH (2020) Gains through selection for grain yield in a winter wheat breeding program. PLoS ONE 15:e0221603. https://doi.org/10.1371/journal.pone.0221603）

Dependent installation: Make sure that the required Python libraries are installed, such as scikit-learn, pandas, numpy, lightgbm, xgboost, etc. You can use pip to install them: pip install scikit-learn pandas numpy lightgbm xgboost memory_profiler.

Run the code: Run the code in a Python environment which will load the data, train the model, evaluate the performance and save the results. Make sure to add if __name__ == '__main__': at the end of the main program to ensure that the code runs correctly in a multi-process environment.

Results Viewing: After running, an Excel file called model_results.xlsx will be generated containing detailed evaluation results for each model as well as the Stacking model.

**adaptiveGS_dl.py**

GenoPhenoDataset: PyTorch custom dataset class.

CNNModel: convolutional model.

MLPModel: simple multilayer perceptual machine model.

TransformerModel: model using Transformer encoder.

train_and_evaluate: train and evaluate function, return model performance metrics.

MLP, CNN, and Transformer models are trained and key metrics such as RMSE, PCC, MAE, R2, and memory usage are calculated.

**SHAP_analysis.py**
get_prediction_data: Load and prepare genotype and phenotype data for SHAP analysis.
run_shap_analysis: Perform SHAP analysis using KernelExplainer, generate feature importance scores and multiple visualizations (bar plot, beeswarm plot, heatmap, dependence plots, decision plot, violin plot).
The script analyzes model interpretability for Stacking models, focusing on feature contributions to wheat yield predictions, and saves results in structured directories.

**permutation_analysis.py** 
get_prediction_data: Load and preprocess genotype/phenotype data for analysis, merging datasets and converting to memory-efficient float32. 
run_permutation_analysis: Perform permutation importance analysis using scikit-learn's permutation_importance. Calculates feature importance via repeated random feature shuffling, generates metrics (mean/std/min/max/median importance), and creates visualizations: 

- Bar plot with error bars (mean ± std importance) 
- Box plot of importance distributions 
- Heatmap of permutation values across repeats 
- Violin plot of importance across repeats 
  Saves results (CSV files and plots) in structured `results` directories. Analyzes Stacking models for wheat yield prediction, evaluating feature impact via model performance degradation on shuffled data.
