# adaptiveGS
## overview
This program implements a Stacking ensemble learning method for model selection based on the PR index (PCC/RMSE)  for  genomic selection. Genomic selection (GS) is a crucial technique in modern molecular breeding, and stacking ensemble learning (SEL) has proven to be a powerful GS approach by integrating multiple base learners (BLs) to capture the complex relationships between genotypes and phenotypes. However, selecting the optimal BLs remains a challenge due to the lack of an effective and unified framework.

**adaptiveGS** is an adaptive and explainable data-driven BL selection strategy designed to enhance GS accuracy. It employs the **PR index**, which combines the **Pearson correlation coefficient (PCC)** and **root mean square error (RMSE)**, to pre-screen and select the top-performing BLs for the stacking GS framework. This strategy ensures that the most suitable BLs are chosen for different species and traits, improving overall prediction performance.

**Citation**
Zhen Yang, Mei Song, Quanrui Rao, Shanghui Zhang, Zhongzheng Zhang, Chenyang Wang, Wenjia Li, Ran Qin, Chunhua Zhao, Yongzhen Wu, Han Sun, Guangchen Liu, Fa Cui, AdaptiveGS: an explainable genomic selection framework based on adaptive stacking ensemble machine learning, 2025,in submiting.

**Suggestions and Problem Feedback**
liuguangchen@ldu.edu.cn

## Key Features
- **Adaptive Base Learner Selection**: Automatically selects the top 3 out of 7 (or user-defined) candidate algorithms based on the PR index.
- **Improved Prediction Accuracy**: Achieved an average prediction accuracy of 0.811 across six traits, with an improvement of 17.9% compared to other GS models.
- **Explainable Model Interpretation**: Utilizes **SHapley Additive exPlanations (SHAP)** to identify key SNPs associated with target traits.
- **Open-Source and User-Friendly**: Available as a Python package for easy integration into genomic selection pipelines.

## Data files
In this section, we introduce 10 trait datasets related to three groups: three traditional genomic selection breeding methods, nine single machine learning methods, and a genomic selection breeding method based on an improved stacking ensemble learning method.
- **Holstein Dairy Cattle Dataset**: The German Holstein cattle dataset consists of 5,024 individual cattle with their genotypes and phenotypes (https://www.g3journal.org/content/5/4/615.supplemental) (Zhang et al. 2015), all of which were genotyped via the Illumina®BovineSNP50 Beadchip, and tagged with 42,551 SNPs. The phenotypes of the dataset used in the present study were standardized (mean= 0, std=1), and the 3 phenotypes corresponding to the Holstein cattle population were milk yield (MY), FPR milk fat percentage (FPR), and somatic cell score (SCS).
- **Loblolly Pine Dataset**: The Loblolly Pine dataset consists of 951 loblolly pine individuals from 61 lineages (https://www.genetics.org/highwire/filestream/412827/field_highwire_adjunct_files/1/FileS1.zip) (Resende et al. 2012). In this study, we selected 2 quantitative traits that are important for tree growth and development, the 4 traits corresponding to total stem height (HT), namely, HTage1, HTage2, HTage3. and HTage6, and crown width (CWAL), corresponding to CWALage2 and CWALage6, for GEBV estimation. The final 6 traits were HTage1, HTage2, HTage3, HTage6, CWALage2, and CWALage6. All 951 individuals in this dataset were genotyped via the Illumina®Iminium chip, and 4,853 SNP loci were retained after quality control.
- **Winter wheat populations**: This dataset contains 10 winter wheat breeding populations adapted to the Pacific Northwest region of the United States (https://doi.org/10.1371/journal.pone.0221603.s013) (Lozada et al. 2020). These populations consisted of 456 varieties with a total of 11,089 labeled SNP loci between 2015 and 2018 in Linder (LND) and Pullman (PUL), Washington, USA. The phenotypic trait for this dataset is grain yield (GY).

## Basic Functions
adaptiveGS.py implements an integrated Stacking-based machine learning framework for model performance evaluation and hyperparameter optimization.
adaptiveGS_dl.py implements a genomic prediction model based on CNNs (Convolutional Neural Networks), MLPs (Multilayer Perceptual Machines), and Transformers, aiming to predict phenotype data from genotype data, in contrast to the Stacking Integrated Learning Model.

## Code files
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
TransformerModel: model using Transformer encoder
train_and_evaluate: train and evaluate function, return model performance metrics.
MLP, CNN, and Transformer models are trained and key metrics such as RMSE, PCC, MAE, R2, and memory usage are calculated.

## Software Requirements
The codes used in this paper were compiled on Python 3.8. The versions of other packages are, specifically:
numpy==1.22.3
pandas==1.4.3
plotly==5.9.0
matplotlib==3.5.1
sklearn==1.1.1
xgboost==1.6.1
seaborn==0.11.2
scipy==1.7.3

python==3.8.0
numpy==1.22.4
pandas==2.0.3
scikit-learn==1.3.0
lightgbm==4.3.0
xgboost==2.0.3
scipy==1.10.1
memory-profiler==0.61.0
torch==1.9.0
psutil==5.9.0
