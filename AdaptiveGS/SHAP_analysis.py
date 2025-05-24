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
import traceback # Import traceback for more detailed error logging

warnings.filterwarnings("ignore")

# Dataset
dataset_paths = {
    "gy_geno": "datasets/Wheat/gy_geno.csv",
    "gy_pheno": "datasets/Wheat/gy_pheno.csv"
}

dataset_config = [
    {"id": "gy", "target": "yield"}
]

model_types = [
    "Stacking",
]

# Define available SHAP explainer types for configuration
SHAP_EXPLAINER_TYPES = [
	"auto",
	# "kernel", 
	# "permutation",
	]

def get_prediction_data(dataset_id, target_col):
    """Load and prepare data for prediction"""
    geno_path = dataset_paths[f"{dataset_id}_geno"]
    pheno_path = dataset_paths[f"{dataset_id}_pheno"]

    if not os.path.exists(geno_path) or not os.path.exists(pheno_path):
        raise FileNotFoundError(f"Dataset files not found for {dataset_id}. Looked for {geno_path} and {pheno_path}")

    geno_data = pd.read_csv(geno_path)
    pheno_data = pd.read_csv(pheno_path)

    # Assuming the first column is an ID column common to both
    id_col_geno = geno_data.columns[0]
    id_col_pheno = pheno_data.columns[0]

    feature_cols = geno_data.columns.tolist()[1:]
    merged_data = pd.merge(
        pheno_data, geno_data,
        left_on=id_col_pheno,
        right_on=id_col_geno
    )
    use_cols = [target_col] + feature_cols
    merged_data = merged_data[use_cols].dropna()
    # Convert to float32 to reduce memory usage
    X_test = merged_data[feature_cols].astype(np.float32)
    y_test = merged_data[target_col].astype(np.float32)

    return X_test, y_test

def run_shap_analysis(model, X_test, task_data, model_type, max_display=20, sample_size=None, explainer_type="auto"):
    """
    Enhanced SHAP analysis supporting different explainer types.

    Parameters:
    -----------
    model : trained model object
    X_test : test features DataFrame
    task_data : str, task identifier
    model_type : str, model type identifier
    max_display : int, maximum number of features to display in plots
    sample_size : int or None, number of samples to analyze (None for all samples)
    explainer_type : str, type of SHAP explainer to use ("auto", "kernel", "permutation")
    """
    if explainer_type not in SHAP_EXPLAINER_TYPES:
        raise ValueError(f"Invalid explainer_type: {explainer_type}. Choose from {SHAP_EXPLAINER_TYPES}.")

    # Include explainer type in the SHAP directory name
    shap_dir = os.path.join("results", task_data, model_type, f"shap_analysis_{explainer_type}")
    os.makedirs(shap_dir, exist_ok=True)

    # Prepare analysis samples
    if sample_size is not None and len(X_test) > sample_size:
        shap_samples = X_test.sample(min(sample_size, len(X_test)), random_state=42)
        print(f"Analyzing {len(shap_samples)} samples (sampled from {len(X_test)} total)")
    else:
        shap_samples = X_test
        print(f"Analyzing all {len(shap_samples)} samples")

    if shap_samples.empty:
        print("No samples to analyze after sampling/filtering. Skipping SHAP analysis.")
        return None, None

    # Determine background data size
    if len(X_test) <= 100:
        background_size = max(1, int(len(X_test) * 0.3)) 
    else:
        background_size = int(len(X_test) * 0.1) 
    background_data = X_test.sample(min(background_size, len(X_test)), random_state=42)
    print(f"Using {len(background_data)} background samples for explainer initialization")

    if background_data.empty:
        print("No background data available. SHAP analysis cannot proceed.")
        return None, None

    # Initialize SHAP Explainer
    explainer = None
    expected_value = None # This will store the baseline prediction

    print(f"Initializing SHAP Explainer (type: {explainer_type})...")
    if explainer_type == "kernel":
        explainer = shap.KernelExplainer(model.predict, background_data)
        expected_value = explainer.expected_value
    elif explainer_type == "permutation":
        explainer = shap.PermutationExplainer(model.predict, background_data)
        expected_value = model.predict(background_data).mean()
    elif explainer_type == "auto":
        explainer = shap.Explainer(model, background_data)
        expected_value = explainer.expected_value 
    else:
        raise ValueError(f"Unexpected explainer_type: {explainer_type}.")

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(shap_samples)
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
    feature_importance_path = os.path.join(shap_dir, f"{task_data}_{explainer_type}_feature_importance.csv")
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"Feature importance saved to {feature_importance_path}")

    # Define common plot title suffix
    plot_title_suffix = f"{model_type} ({explainer_type} SHAP)"

    # 1. Bar Plot of Feature Importance
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_samples,
                     plot_type="bar",
                     max_display=max_display,
                     show=False)
    plt.title(f"{task_data} {plot_title_suffix} Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "1_feature_importance_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Beeswarm Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, shap_samples,
                     max_display=max_display,
                     show=False)
    plt.title(f"{task_data} {plot_title_suffix} SHAP Value Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "2_beeswarm_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap of SHAP values (Top N features)
    if shap_samples.shape[0] > 1: 
        plt.figure(figsize=(15, 10))
        top_features = feature_importance['Feature'].head(max_display).tolist()
        if top_features:
            # Filter shap_values and shap_samples to include only top features
            feature_indices = [list(shap_samples.columns).index(f) for f in top_features]
            shap_values_top = shap_values[:, feature_indices]
            sns.heatmap(shap_values_top,
                        xticklabels=top_features,
                        yticklabels=False,
                        cmap='RdBu',
                        center=0,
                        cbar_kws={'label': 'SHAP Value'})
            plt.title(f"{task_data} {plot_title_suffix} SHAP Values Heatmap (Top {max_display} Features)")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, "3_shap_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print(f"Skipping Heatmap: No top features found for max_display={max_display}.")
    else:
        print("Skipping Heatmap: Not enough samples (need > 1) to create a meaningful heatmap.")


    # 4. Individual Feature Dependence Plots (Top 5)
    top_5_features = feature_importance['Feature'].head(5).tolist()
    if top_5_features:
        for i, feature in enumerate(top_5_features):
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature, shap_values, shap_samples,
                interaction_index=None, 
                show=False
            )
            plt.title(f"{task_data} {plot_title_suffix} {feature} SHAP Dependence")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"4_{i+1}_dependence_{feature}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    else:
        print("Skipping Dependence Plots: No features to display.")

    # 5. SHAP Decision Plot
    # Decision plot requires expected_value and makes most sense with explainers that explicitly provide it.
    if expected_value is not None:
        # Limit number of samples for decision plot for readability
        num_decision_samples = min(200, len(shap_samples)) 
        if num_decision_samples > 0:
            plt.figure(figsize=(12, 8))
            shap.decision_plot(expected_value,
                              shap_values[:num_decision_samples],
                              shap_samples.iloc[:num_decision_samples],
                              feature_names=shap_samples.columns.tolist(),
                              show=False)
            plt.title(f"{task_data} {plot_title_suffix} Decision Plot (First {num_decision_samples} Samples)")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, "5_decision_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Skipping Decision Plot: Not enough samples to plot.")
    else:
        print("Skipping Decision Plot: Expected value could not be reliably determined or is not applicable for this explainer type.")

    # 6. Violin Plot of SHAP Values (for top features)
    top_features = feature_importance['Feature'].head(max_display).tolist()
    if top_features and shap_samples.shape[0] > 1: # Violin plot needs more than one sample
        plt.figure(figsize=(15, 8))
        shap_df = pd.DataFrame(shap_values, columns=shap_samples.columns)
        # Select only top features for melting and plotting
        shap_melted = pd.melt(shap_df[top_features], var_name='Feature', value_name='SHAP Value')
        sns.violinplot(x='Feature', y='SHAP Value', data=shap_melted, inner='quartile')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"{task_data} {plot_title_suffix} SHAP Values Distribution (Top {max_display} Features)")
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, "6_violin_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Skipping Violin Plot: Not enough samples (need > 1) or features to create a meaningful violin plot.")

    # Save raw SHAP values
    shap_values_df = pd.DataFrame(
        shap_values,
        columns=shap_samples.columns,
        index=shap_samples.index
    )
    shap_values_path = os.path.join(shap_dir, f"{task_data}_{explainer_type}_shap_values.csv")
    shap_values_df.to_csv(shap_values_path)
    print(f"Raw SHAP values saved to {shap_values_path}")

    print(f"Analysis completed. Results saved in: {shap_dir}")
    return feature_importance, shap_values_df

if __name__ == "__main__":

    print(f"Starting analysis at UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    for dataset in dataset_config:
        dataset_id = dataset["id"]
        target_col = dataset["target"]

        for model_type in model_types:
            # Iterate through each defined SHAP explainer type
            for explainer_type in SHAP_EXPLAINER_TYPES:
                print(f"\n--- Processing {dataset_id} - {target_col} with {model_type} using {explainer_type} explainer ---")

                task_data = f"{dataset_id}_{target_col}"
                model_path = os.path.join(
                    "checkpoints",
                    task_data,
                    model_type,
                    f"{model_type}.joblib"
                )

                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}. Skipping this combination.")
                    continue

                try:
                    # Load model and data
                    model = joblib.load(model_path)
                    X_test, y_test = get_prediction_data(dataset_id, target_col)

                    # Ensure X_test has column names for SHAP plotting
                    if not isinstance(X_test, pd.DataFrame):
                        X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])

                    # Run SHAP analysis
                    feature_importance, shap_values = run_shap_analysis(
                        model=model,
                        X_test=X_test,
                        task_data=task_data,
                        model_type=model_type,
                        sample_size=None,  # Set to an integer like 500 to sample, None to use all
                        explainer_type=explainer_type 
                    )

                except Exception as e:
                    print(f"Error processing {task_data} - {model_type} with {explainer_type} explainer: {e}")
                    traceback.print_exc() # Print full traceback for detailed debugging
                    continue

    print(f"\nAnalysis finished at UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
