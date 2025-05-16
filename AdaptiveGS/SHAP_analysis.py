import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from datetime import datetime
import random
import warnings
warnings.filterwarnings("ignore")

# Dataset 
DATASET_PATHS = {
    "gy_geno": "datasets/Wheat/gy_geno.csv",
    "gy_pheno": "datasets/Wheat/gy_pheno.csv"
}

DATASET_CONFIG = [
    {"id": "gy", "target": "yield"}
]

MODEL_TYPES = [
    "Stacking",
]

def get_prediction_data(dataset_id, target_col):
    """Load and prepare data for prediction"""
    geno_path = DATASET_PATHS[f"{dataset_id}_geno"]
    pheno_path = DATASET_PATHS[f"{dataset_id}_pheno"]
    
    geno_data = pd.read_csv(geno_path)
    pheno_data = pd.read_csv(pheno_path)
    
    feature_cols = geno_data.columns.tolist()[1:]
    merged_data = pd.merge(
        pheno_data, geno_data,
        left_on=pheno_data.columns[0], 
        right_on=geno_data.columns[0]
    )
    use_cols = [target_col] + feature_cols
    merged_data = merged_data[use_cols].dropna()
    # Convert to float32 to reduce memory usage
    X_test = merged_data[feature_cols].astype(np.float32)
    y_test = merged_data[target_col].astype(np.float32)
    
    return X_test, y_test

def run_shap_analysis(model, X_test, task_data, model_type, max_display=20, sample_size=None):
    """
    Enhanced SHAP analysis using KernelExplainer
    
    Parameters:
    -----------
    model : trained model object
    X_test : test features DataFrame
    task_data : str, task identifier
    model_type : str, model type identifier
    max_display : int, maximum number of features to display in plots
    sample_size : int or None, number of samples to analyze (None for all samples)
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    shap_dir = os.path.join("results", task_data, model_type, f"shap_analysis")
    os.makedirs(shap_dir, exist_ok=True)

    # Prepare analysis samples
    if sample_size is not None:
        shap_samples = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    else:
        shap_samples = X_test
    print(f"Analyzing {len(shap_samples)} samples")

    # Determine background data size
    if len(X_test) < 300:
        background_size = 100 # int(len(X_test) * 0.3)
    else:
        background_size = 80 # int(len(X_test) * 0.1)
    background_data = X_test.sample(background_size, random_state=42)
    print(f"Using {background_size} background samples")

    # Initialize KernelExplainer
    print("Initializing KernelExplainer...")
    explainer = shap.KernelExplainer(model.predict, background_data)

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(shap_samples,n_jobs=-1)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Calculate and save feature importance
    feature_importance = pd.DataFrame({
        'Feature': shap_samples.columns,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0),
        'Mean_SHAP': shap_values.mean(axis=0),
        'Std_SHAP': shap_values.std(axis=0),
        'Min_SHAP': shap_values.min(axis=0),
        'Max_SHAP': shap_values.max(axis=0),
        'Median_SHAP': np.median(shap_values, axis=0)
    })
    feature_importance = feature_importance.sort_values('Mean_Abs_SHAP', ascending=False)
    feature_importance.to_csv(os.path.join(shap_dir, f"{task_data}_feature_importance.csv"), index=False)

    # 1. Bar Plot of Feature Importance
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_samples, 
                     plot_type="bar", 
                     max_display=max_display,
                     show=False)
    plt.title(f"{task_data} {model_type} Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "1_feature_importance_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Beeswarm Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, shap_samples, 
                     max_display=max_display,
                     show=False)
    plt.title(f"{task_data} {model_type} SHAP Value Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "2_beeswarm_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap of SHAP values
    plt.figure(figsize=(15, 10))
    top_features = feature_importance['Feature'].head(max_display).tolist()
    shap_values_top = shap_values[:, [list(shap_samples.columns).index(f) for f in top_features]]
    sns.heatmap(shap_values_top, 
                xticklabels=top_features,
                yticklabels=False,
                cmap='RdBu',
                center=0)
    plt.title(f"{task_data} {model_type} SHAP Values Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "3_shap_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Individual Feature Dependence Plots (Top 5)
    top_5_features = feature_importance['Feature'].head(5).tolist()
    for i, feature in enumerate(top_5_features):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, shap_values, shap_samples,
            interaction_index=None,
            show=False
        )
        plt.title(f"{task_data} {model_type} {feature} SHAP Dependence")
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f"4_{i+1}_dependence_{feature}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 5. SHAP Decision Plot
    plt.figure(figsize=(12, 8))
    shap.decision_plot(explainer.expected_value, 
                      shap_values[:200],  # First 10 samples
                      shap_samples.iloc[:200],
                      feature_names=shap_samples.columns.tolist(),
                      show=False)
    plt.title(f"{task_data} {model_type} Decision Plot (First 10 Samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "5_decision_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Violin Plot of SHAP Values
    plt.figure(figsize=(15, 8))
    shap_df = pd.DataFrame(shap_values, columns=shap_samples.columns)
    shap_melted = pd.melt(shap_df[top_features])
    sns.violinplot(x='variable', y='value', data=shap_melted)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"{task_data} {model_type} SHAP Values Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "6_violin_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save raw SHAP values
    shap_values_df = pd.DataFrame(
        shap_values, 
        columns=shap_samples.columns,
        index=shap_samples.index
    )
    shap_values_df.to_csv(os.path.join(shap_dir, f"{task_data}_shap_values.csv"))

    print(f"Analysis completed. Results saved in: {shap_dir}")
    return feature_importance, shap_values_df

if __name__ == "__main__":
    """Main execution function"""
    print(f"Starting analysis at UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for dataset in DATASET_CONFIG:
        dataset_id = dataset["id"]
        target_col = dataset["target"]
        
        for model_type in MODEL_TYPES:
            print(f"\nProcessing {dataset_id} - {target_col} with {model_type}")
            
            task_data = f"{dataset_id}_{target_col}"
            model_path = os.path.join(
                "checkpoints",
                task_data,
                model_type,
                f"{model_type}.joblib"
            )
            
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
                
            try:
                # Load model and data
                model = joblib.load(model_path)
                X_test, y_test = get_prediction_data(dataset_id, target_col)
                
                # Run SHAP analysis
                feature_importance, shap_values = run_shap_analysis(
                    model=model,
                    X_test=X_test,
                    task_data=task_data,
                    model_type=model_type,
                    sample_size=None  # Use all samples
                )
                
            except Exception as e:
                print(f"Error processing {task_data} - {model_type}: {str(e)}")
                continue