import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.inspection import permutation_importance
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
# Dataset and model configurations
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


def run_permutation_analysis(model, X_test, y_test, task_data, model_type, n_repeats=10, max_display=20):
    """
    Enhanced permutation importance analysis
    
    Parameters:
    -----------
    model : trained model object
    X_test : test features DataFrame
    y_test : test target Series
    task_data : str, task identifier
    model_type : str, model type identifier
    n_repeats : int, number of times to repeat permutation
    max_display : int, maximum number of features to display in plots
    """
    # Create output directory
    perm_dir = os.path.join("results", task_data, model_type, "permutation_analysis")
    os.makedirs(perm_dir, exist_ok=True)

    print("Calculating permutation importance...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )

    # Calculate and save feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_Importance': result.importances_mean,
        'Std_Importance': result.importances_std,
        'Min_Importance': np.min(result.importances, axis=1),
        'Max_Importance': np.max(result.importances, axis=1),
        'Median_Importance': np.median(result.importances, axis=1)
    })
    feature_importance = feature_importance.sort_values('Mean_Importance', ascending=False)
    feature_importance.to_csv(os.path.join(perm_dir, f"{task_data}_feature_importance.csv"), index=False)

    # Save raw permutation values
    all_permutation_values = pd.DataFrame(
        result.importances,
        columns=[f'repeat_{i}' for i in range(n_repeats)],
        index=X_test.columns
    )
    all_permutation_values.to_csv(os.path.join(perm_dir, f"{task_data}_permutation_values.csv"))

    # 1. Bar Plot of Feature Importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance['Feature'].head(max_display)
    plt.barh(range(len(top_features)), 
             feature_importance['Mean_Importance'].head(max_display),
             yerr=feature_importance['Std_Importance'].head(max_display),
             capsize=5)
    plt.yticks(range(len(top_features)), top_features)
    plt.title(f"{task_data} {model_type} Feature Importance (Permutation)")
    plt.xlabel("Mean Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(perm_dir, "1_feature_importance_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Box Plot of Feature Importance
    plt.figure(figsize=(12, 8))
    data_to_plot = []
    labels = []
    for feature in top_features:
        idx = list(X_test.columns).index(feature)
        data_to_plot.append(result.importances[idx])
        labels.append(feature)
    
    plt.boxplot(data_to_plot, labels=labels, vert=False)
    plt.title(f"{task_data} {model_type} Permutation Importance Distribution")
    plt.xlabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(perm_dir, "2_importance_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap of Permutation Values
    plt.figure(figsize=(15, 10))
    top_features_data = all_permutation_values.loc[top_features]
    sns.heatmap(top_features_data, 
                cmap='RdBu',
                center=0,
                xticklabels=True,
                yticklabels=True)
    plt.title(f"{task_data} {model_type} Permutation Values Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(perm_dir, "3_permutation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Violin Plot of Permutation Values
    plt.figure(figsize=(15, 8))
    data_melted = pd.melt(all_permutation_values.loc[top_features])
    sns.violinplot(x='variable', y='value', data=data_melted)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"{task_data} {model_type} Permutation Values Distribution")
    plt.xlabel("Repeat")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(perm_dir, "4_violin_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Analysis completed. Results saved in: {perm_dir}")
    return feature_importance, all_permutation_values



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
                
                # Run permutation analysis
                feature_importance, perm_values = run_permutation_analysis(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    task_data=task_data,
                    model_type=model_type
                )
                
            except Exception as e:
                print(f"Error processing {task_data} - {model_type}: {str(e)}")
                continue