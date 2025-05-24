import numpy as np
import pandas as pd
import os
import joblib
import time
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, LinearRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from memory_profiler import memory_usage
from sklearn.preprocessing import StandardScaler 
import gc
import warnings
warnings.filterwarnings('ignore')

# --------------------------- Global Configuration ---------------------------
# Dataset paths
dataset_paths = {
    "gy_geno": "datasets/Wheat/gy_geno.csv", # "{dataset_id}_pheno/geno":"data_file_path"
    "gy_pheno": "datasets/Wheat/gy_pheno.csv"
}

# Phenotype dictionary mapping dataset_id_pheno to target columns
pheno_dict = {
    'gy_pheno': ['yield'] # "{dataset_id}_pheno":['phenotype1','phenotype2',...]
}

# --------------------------- Data Preprocessing ---------------------------

def get_data(dataset_id_t, y_temp_col):
    """
    Safely loads, cleans, and standardizes phenotype data.
    Uses specified nrows and usecols for quick testing.
    Returns train/test sets as NumPy arrays (X unscaled, y scaled)
    and the fitted y scaler.
    """
    try:
        geno_key = dataset_id_t + '_geno'
        pheno_key = dataset_id_t + '_pheno'

        if geno_key not in dataset_paths or pheno_key not in dataset_paths:
             raise KeyError(f"Dataset paths not found for {dataset_id_t}")

        geno_path = dataset_paths[geno_key]
        pheno_path = dataset_paths[pheno_key]

        if not os.path.exists(geno_path):
             raise FileNotFoundError(f"Genotype file not found at {geno_path}")
        if not os.path.exists(pheno_path):
             raise FileNotFoundError(f"Phenotype file not found at {pheno_path}")

        geno_data = pd.read_csv(geno_path)
        pheno_data = pd.read_csv(pheno_path)                

        # The first column is the sample identifier in both files
        sample_id_col_pheno = pheno_data.columns[0]
        sample_id_col_geno = geno_data.columns[0]

        concat_data = pd.merge(
            pheno_data,
            geno_data,
            left_on=sample_id_col_pheno,
            right_on=sample_id_col_geno
        )

        x_col = [col for col in geno_data.columns if col != sample_id_col_geno]
        use_col = [y_temp_col] + x_col
        use_col_existing = [col for col in use_col if col in concat_data.columns]
        concat_data = concat_data[use_col_existing].dropna()

        if concat_data.empty or y_temp_col not in concat_data.columns or not x_col:
             raise ValueError(f"After merging, subsetting and dropping NaNs, no valid data remains for {dataset_id_t}_{y_temp_col} or columns are missing.")

        # Separate features (X) and target (y)
        X = concat_data[x_col] 
        y = concat_data[y_temp_col] 

        if X.shape[0] < 2 or y.shape[0] < 2:
             raise ValueError(f"Not enough data points after preprocessing for {dataset_id_t}_{y_temp_col}. Need at least 2 samples.")


        # Split data into train and test sets 
        X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )

        # Data Standardization: Only standardize y (phenotype)
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train_s.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test_s.values.reshape(-1, 1)).flatten()

        # Convert X  to NumPy arrays (float32)
        X_train_np = X_train_df.values.astype('float32')
        X_test_np = X_test_df.values.astype('float32')

        y_train_unscaled = y_train_s.values.astype('float32')
        y_test_unscaled = y_test_s.values.astype('float32')


        print(f"Data loaded, cleaned, and phenotype standardized. X_train shape: {X_train_np.shape}, y_train_scaled shape: {y_train_scaled.shape}")

        # Return unscaled X arrays, scaled y arrays, unscaled original y arrays, and the fitted y scaler
        return X_train_np, X_test_np, y_train_scaled, y_test_scaled, y_test_unscaled, scaler_y

    except (KeyError, FileNotFoundError, ValueError, Exception) as e:
        # Re-raise as RuntimeError to be caught in the main loop
        raise RuntimeError(f"Data loading or processing failed for {dataset_id_t}_{y_temp_col}: {str(e)}")

# --------------------------- Model Training and Evaluation ---------------------------

def save_model(model, dataset_id, model_name, save_path):
    """
    Saves a trained model to a specified path.

    Args:
        model: The trained model.
        dataset_id: Identifier for the dataset (e.g., 'gy_yield').
        model_name: Name of the model (e.g., 'LightGBM_Selected').
        save_path: Base directory to save models.
    """
    # Create save path
    save_dir = os.path.join(save_path, dataset_id,f'{model_name}') 
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, f'{model_name}.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    return model_path

def pcc_scorer_optuna(y_true, y_pred):
    """
    Pearson Correlation Coefficient scorer for Optuna (maximize correlation).
    Expects scaled true and predicted values from OptunaSearchCV's CV.
    """
    corr, _ = pearsonr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0


def evaluate_model(model, X_train, y_train_scaled, X_test, y_test_scaled, y_test_unscaled, scaler_y, model_name=None):
    """
    Fits the model on scaled training data, measures fit time/memory,
    predicts on test data, measures predict time, inverse transforms predictions,
    and calculates standard regression metrics on unscaled values.
    Returns a dictionary of results.
    """
    gc.collect() # Clean up memory before measurement

    # Measure Fit Time and Memory
    start_fit = time.time()
    # memory_usage can sometimes return a list or single value
    mem_usage = memory_usage((model.fit, (X_train, y_train_scaled)), max_usage=True)
    fit_time = time.time() - start_fit
    max_mem_usage = max(mem_usage) if isinstance(mem_usage, (list, tuple)) else mem_usage

    # Measure Prediction Time
    start_pred = time.time()
    y_pred_scaled = model.predict(X_test)
    pred_time = time.time() - start_pred

    # Ensure predictions have the correct shape for inverse_transform (2D)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    # Inverse transform predictions using the fitted scaler
    y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled).flatten()


    # Calculate Metrics on UNCALED data
    # Use y_test_unscaled directly, which was stored during get_data
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    pcc, _ = pearsonr(y_test_unscaled, y_pred_unscaled)
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)

    # Handle potential NaN Pearson correlation (e.g., if y_true or y_pred is constant)
    pcc = pcc if not np.isnan(pcc) else 0.0


    results = {
        'Model': model_name if model_name else type(model).__name__,
        'Training Time (s)': fit_time,
        'Max Memory Usage (MiB)': max_mem_usage,
        'Prediction Time (s)': pred_time,
        'RMSE': rmse,
        'PCC': pcc,
        'MAE': mae,
        'R2': r2
    }
    return results


# normalization function
def normalize_rmse(rmse_list):
    """Normalizes a list of RMSE values to a 0-1 range."""
    # Avoid modifying the original list
    rmse_values = np.array(rmse_list)

    # Handle edge cases for normalization
    if len(rmse_values) == 0:
        return []
    # Adjust the boundary to prevent the NRMSE from fluctuating excessively.
    min_rmse = np.min(rmse_values)-1
    max_rmse = np.max(rmse_values)+1

    # If all RMSE values are the same, return 1.0 for all
    if np.isclose(min_rmse, max_rmse):
        return [1.0] * len(rmse_values)

    # Apply min-max normalization
    normalized = (rmse_values - min_rmse) / (max_rmse - min_rmse)
    # To prevent the PR index from varying too significantly due to an excessively small NRMSE, 
    # we can add 0.5 to map it to the interval [0.5, 1.5].
    # normalized = (rmse_values - min_rmse) / (max_rmse - min_rmse) + 0.5 
    return normalized.tolist()


# Updated model configurations with proper Optuna distributions and regularization parameters
def get_model_configs():
    """Returns a dictionary of base models and their Optuna parameter distributions."""
    return {
        'SVR': {
            'model': SVR(),
            'param_distributions': {
                'C': optuna.distributions.FloatDistribution(0.1, 500, log=True),
                'gamma': optuna.distributions.FloatDistribution(1e-4, 1, log=True),
                'kernel': optuna.distributions.CategoricalDistribution(['linear', 'rbf']), # Exclude poly for stability/speed
                'epsilon': optuna.distributions.FloatDistribution(0.01, 0.1)
            }
        },
        'Kernel Ridge': {
            'model': KernelRidge(),
            'param_distributions': {
                'alpha': optuna.distributions.FloatDistribution(0.01, 10, log=True),
                'kernel': optuna.distributions.CategoricalDistribution(['linear', 'rbf']),
                'gamma': optuna.distributions.FloatDistribution(1e-4, 1, log=True)
            }
        },
        'ElasticNet': {
            'model': ElasticNet(random_state=42),
            'param_distributions': {
                'alpha': optuna.distributions.FloatDistribution(0.01, 1, log=True),
                'l1_ratio': optuna.distributions.FloatDistribution(0.01, 1),
                'max_iter': optuna.distributions.IntDistribution(1000, 10000)
            }
        },
        'LightGBM': {
            'model': lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1),
            'param_distributions': {
                'num_leaves': optuna.distributions.IntDistribution(20, 150),
                'max_depth': optuna.distributions.IntDistribution(3, 20),
                'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                'n_estimators': optuna.distributions.IntDistribution(50, 1000),
                'reg_alpha': optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
                'reg_lambda': optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
                'min_child_samples': optuna.distributions.IntDistribution(5, 100),
                'subsample': optuna.distributions.FloatDistribution(0.5, 1.0),
                'colsample_bytree': optuna.distributions.FloatDistribution(0.5, 1.0)
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
            'param_distributions': {
                'max_depth': optuna.distributions.IntDistribution(3, 20),
                'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                'n_estimators': optuna.distributions.IntDistribution(50, 1000),
                'gamma': optuna.distributions.FloatDistribution(1e-8, 1.0, log=True),
                'subsample': optuna.distributions.FloatDistribution(0.5, 1),
                'colsample_bytree': optuna.distributions.FloatDistribution(0.5, 1),
                'reg_alpha': optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
                'reg_lambda': optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
                'min_child_weight': optuna.distributions.IntDistribution(1, 20)
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'param_distributions': {
                'n_estimators': optuna.distributions.IntDistribution(50, 1000),
                'max_depth': optuna.distributions.IntDistribution(3, 20),
                'min_samples_split': optuna.distributions.IntDistribution(2, 20),
                'min_samples_leaf': optuna.distributions.IntDistribution(1, 10),
                'max_features': optuna.distributions.CategoricalDistribution(['sqrt', 'log2', None]),
                'bootstrap': optuna.distributions.CategoricalDistribution([True, False]),
                'max_samples': optuna.distributions.FloatDistribution(0.5, 1.0)
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'param_distributions': {
                'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                'n_estimators': optuna.distributions.IntDistribution(50, 1000),
                'max_depth': optuna.distributions.IntDistribution(3, 20),
                'min_samples_split': optuna.distributions.IntDistribution(2, 20),
                'min_samples_leaf': optuna.distributions.IntDistribution(1, 10),
                'subsample': optuna.distributions.FloatDistribution(0.5, 1.0),
                'max_features': optuna.distributions.FloatDistribution(0.5, 1.0),
            }
        }
    }


# --------------------------- Main Execution ---------------------------

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    save_path = 'checkpoints' # Path of checkpoints and results 
    
    # Get unique dataset identifiers (e.g., 'gy') from pheno_dict keys
    temp_dataset_ids = list({k.replace('_pheno', '') for k in pheno_dict.keys()})

    n_trials_optuna = 20        # Number of Optuna trials per model
    cv_folds_optuna = 10        # CV folds for Optuna
    base_select_model_n = 3     # Top N models for stacking
    cv_folds_stacking = 10      # CV folds for stacking
    
    
    for dataset_id_t in temp_dataset_ids:
        # Iterate over phenotypes for this dataset
        pheno_key = dataset_id_t + '_pheno'
        if pheno_key not in pheno_dict:
             print(f"Skipping dataset {dataset_id_t}: No phenotype info found in pheno_dict.")
             continue

        for y_temp_col in pheno_dict[pheno_key]:
            model_configs = get_model_configs()
            task_results = [] # List to collect results for the current task
            optimized_models = {} # Store selected base models for stacking
            task_data_id = f'{dataset_id_t}_{y_temp_col}' # Unique ID for this task (e.g., gy_yield)

            print(f"\n--- Processing Task: {task_data_id} ---")

            try:
                # Load and split data, get scaler
                X_train, X_test, y_train_scaled, y_test_scaled, y_test_unscaled, scaler_y = get_data(dataset_id_t, y_temp_col)

                # Check if data splitting resulted in meaningful splits
                if len(X_train) == 0 or len(X_test) == 0:
                     raise ValueError("Data splitting resulted in empty train or test set.")
                if len(np.unique(y_train_scaled)) < 2 or len(np.unique(y_test_scaled)) < 2:
                     warnings.warn(f"Target variable {y_temp_col} has fewer than 2 unique values in train or test set after scaling. Correlation might be ill-defined.")


            except RuntimeError as e:
                print(f"Error loading data for {task_data_id}: {e}")
                continue # Skip to the next task
            except ValueError as e:
                 print(f"Error with data for {task_data_id}: {e}")
                 continue # Skip to the next task


            # --------------------------- Base Model Tuning and Evaluation ---------------------------
            print(f"\nStarting evaluation and tuning for base models ({task_data_id})...")
            base_model_eval_results = [] # Store results for plotting/selection before stacking

            for model_name, config in model_configs.items():
                print(f"\n=== Processing Model: {model_name} ===")

                base_model_instance = config['model']
                param_dist = config['param_distributions']

                # 1. Perform Hyperparameter Tuning with Optuna
                print(f"Starting Optuna Tuning ({n_trials_optuna} trials, {cv_folds_optuna} folds)...")
                search = OptunaSearchCV(
                    base_model_instance,
                    param_dist,
                    cv=KFold(n_splits=cv_folds_optuna, shuffle=True, random_state=42),
                    n_jobs=-1, # Use all available cores for tuning CV folds
                    n_trials=n_trials_optuna,
                    scoring=make_scorer(pcc_scorer_optuna, greater_is_better=True), # Maximize PCC on scaled CV data
                    random_state=42,
                    timeout=3600*3 # 3 hour timeout per model tuning
                )

                gc.collect()
                try:
                    # OptunaSearchCV will fit on scaled data internally during CV
                    tuning_start_time = time.time()
                    search.fit(X_train, y_train_scaled)
                    tuning_duration = time.time() - tuning_start_time

                    best_model_after_tuning = search.best_estimator_
                    best_params = search.best_params_
                    print(f"Optuna Tuning finished in {tuning_duration:.2f}s. Best params: {best_params}")

                    # 2. Evaluate the BEST model found by Optuna (refitting on full training data inside evaluate_model)
                    print("Evaluating Tuned Model (refitting on full training data)...")
                    tuned_results = evaluate_model(
                        best_model_after_tuning, X_train, y_train_scaled, X_test, y_test_scaled, y_test_unscaled, scaler_y, model_name=model_name
                    )
                    tuned_results['Best Params'] = str(best_params) # Store params as string
                    tuned_results['Dataset'] = task_data_id
                    print(f"Tuned Model {model_name} Performance: PCC={tuned_results['PCC']:.4f}, RMSE={tuned_results['RMSE']:.4f}, Train Time={tuned_results['Training Time (s)']:.2f}s, Pred Time={tuned_results['Prediction Time (s)']:.4f}s")

                    # Select the tuned model as the candidate for stacking and final results
                    selected_model_instance = best_model_after_tuning
                    selected_results = tuned_results

                except Exception as e:
                    print(f"Error during Optuna tuning or evaluation for {model_name}: {e}")
                    # If tuning fails or errors, just evaluate the base model with default params
                    print(f"Evaluating Base Model (default params) due to tuning error...")
                    base_results = evaluate_model(
                        base_model_instance, X_train, y_train_scaled, X_test, y_test_scaled, y_test_unscaled, scaler_y, model_name=model_name
                    )
                    base_results['Best Params'] = 'Primary' # Indicate default params
                    base_results['Dataset'] = task_data_id
                    print(f"Base Model {model_name} Performance: PCC={base_results['PCC']:.4f}, RMSE={base_results['RMSE']:.4f}, Train Time={base_results['Training Time (s)']:.2f}s, Pred Time={base_results['Prediction Time (s)']:.4f}s")

                    selected_model_instance = base_model_instance
                    selected_results = base_results

                # Store the selected model and its results
                optimized_models[model_name] = selected_model_instance
                base_model_eval_results.append(selected_results) # Add to list for base model comparison/stacking selection

                # Save the selected model (which is either the best tuned or the base model)
                save_model_name = f'{model_name}'
                save_model(selected_model_instance, task_data_id, save_model_name, save_path)

                gc.collect() # Clean up memory after processing each model


            # --------------------------- Prepare for Stacking ---------------------------
            if not base_model_eval_results:
                 print(f"\nNo base models were successfully evaluated for stacking ({task_data_id}). Skipping stacking.")
                 continue

            # Calculate Normalized RMSE and PR index for selected base models
            rmse_list = [r['RMSE'] for r in base_model_eval_results]
            normalized_rmse = normalize_rmse(rmse_list)

            # Add calculated metrics to base model results
            for i, result in enumerate(base_model_eval_results):
                result['Normalized RMSE'] = normalized_rmse[i]
                # Calculate PR index using the normalized RMSE
                # Handle division by zero if normalized RMSE is 0 (shouldn't happen if normalize_rmse is correct, but safety check)
                result['PR index'] = result['PCC'] / result['Normalized RMSE'] if result['Normalized RMSE'] != 0 else float('inf')

            # Select top models for Stacking (using the models evaluated and stored in optimized_models)
            base_results_df = pd.DataFrame(base_model_eval_results)
            # Sort by PR index descending
            base_results_df_sorted = base_results_df.sort_values(by='PR index', ascending=False)

            # Select top models based on sorted results. Get model instances from optimized_models
            selected_estimators_list = []
            for index, row in base_results_df_sorted.iterrows():
                 model_name = row['Model'] 
                 original_model_name = model_name.split(" (")[0]
                 if original_model_name in optimized_models:
                    selected_estimators_list.append((original_model_name, optimized_models[original_model_name]))
                 else:
                     print(f"Warning: Could not find model instance for {original_model_name} in optimized_models.")
                 model_name = row['Model']
                 if len(selected_estimators_list) >= max(base_select_model_n, 1):
                     break

            if len(selected_estimators_list) < base_select_model_n:
                print(f"\nSkipping Stacking Model: Less than {base_select_model_n} selected base models available.")
                # Add base model results to final results before skipping stacking
                task_results.extend(base_model_eval_results)
            else:
                print(f"\nSelected models for Stacking: {[name for name, _ in selected_estimators_list]}")

                # Create Stacking Regressor
                stack = StackingRegressor(
                    estimators=selected_estimators_list,
                    final_estimator=LinearRegression(), # Simple linear model as final layer, tune its params
                    cv=cv_folds_stacking, # Use defined folds for stacking CV
                    n_jobs=-1
                )
                stack_param_distributions = {}
                for name, _ in selected_estimators_list:
                    # Find the original config for the model name
                    if name in get_model_configs():
                        base_params = get_model_configs()[name]['param_distributions']
                        for param, dist in base_params.items():
                            # Parameter names for estimators inside StackingRegressor are prefixed
                            stack_param_distributions[f"{name}__{param}"] = dist


                print(f"\nStarting Optuna Tuning for Stacking Model ({n_trials_optuna} trials, {cv_folds_optuna} folds)...")
                # Use the same number of trials/folds for stacking tuning as base models for consistency
                stack_search = OptunaSearchCV(
                    stack,
                    stack_param_distributions,
                    n_trials=n_trials_optuna,
                    cv=KFold(n_splits=cv_folds_optuna, shuffle=True, random_state=42), # Use Optuna folds here
                    scoring=make_scorer(pcc_scorer_optuna, greater_is_better=True), # Maximize PCC on scaled CV data
                    n_jobs=-1,
                    random_state=42,
                    timeout=3*3600 # 3 hour timeout
                )

                gc.collect()
                try:
                    # Stacking search fits the ensemble during CV on scaled data
                    stack_tuning_start_time = time.time()
                    stack_search.fit(X_train, y_train_scaled)
                    stack_tuning_duration = time.time() - stack_tuning_start_time

                    best_stack_model = stack_search.best_estimator_
                    best_stack_params = stack_search.best_params_
                    print(f"Stacking Optuna Tuning finished in {stack_tuning_duration:.2f}s. Best params: {best_stack_params}")

                    # 3. Evaluate the BEST Stacking model found by Optuna
                    print("Evaluating Tuned Stacking Model (refitting on full training data)...")
                    stack_results = evaluate_model(
                        best_stack_model, X_train, y_train_scaled, X_test, y_test_scaled, y_test_unscaled, scaler_y, model_name='Stacking'
                    )
                    stack_results['Best Params'] = str(best_stack_params) # Store params as string
                    stack_results['Dataset'] = task_data_id
                    print(f"Stacking Model Performance: PCC={stack_results['PCC']:.4f}, RMSE={stack_results['RMSE']:.4f}, Train Time={stack_results['Training Time (s)']:.2f}s, Pred Time={stack_results['Prediction Time (s)']:.4f}s")

                    # Add Stacking results to the list
                    task_results.extend(base_model_eval_results) 
                    task_results.append(stack_results)          

                    # Save the stacking model
                    save_model(best_stack_model, task_data_id, 'Stacking_model', save_path)

                except Exception as e:
                    print(f"Error during Stacking tuning or evaluation: {e}")
                    # If stacking fails, just include the base model results
                    print("Skipping Stacking results due to error.")
                    task_results.extend(base_model_eval_results)


            # --------------------------- Final Results Compilation ---------------------------
            # Calculate Normalized RMSE and PR index for ALL models in task_results
            if task_results:
                final_rmse_list = [r['RMSE'] for r in task_results]
                final_normalized_rmse = normalize_rmse(final_rmse_list)
                results_df = pd.DataFrame(task_results)

                # Reorder columns to match the requested format
                results_df = results_df[[
                    'Model', 'Dataset', 'Training Time (s)', 'Prediction Time (s)',
                    'Max Memory Usage (MiB)', 'RMSE', 'PCC', 'MAE', 'R2',
                    'PR index', 'Best Params'
                ]]
                results_df = results_df.sort_values(by=['Dataset', 'PR index'], ascending=[True, False])
                # Save results for the current task
                results_path = os.path.join(save_path, task_data_id, 'ML_performance.xlsx')
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                results_df.to_excel(results_path, index=False)
                print(f"\nResults for {task_data_id} saved to: {results_path}")
            else:
                print(f"No results generated for task {task_data_id}.")


            gc.collect() # Final cleanup after processing a task

    print("\n--- Processing Complete ---")
