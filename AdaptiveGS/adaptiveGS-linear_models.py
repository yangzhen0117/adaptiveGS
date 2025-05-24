import os
os.environ['R_HOME'] = r'D:\R\R-4.2.3' # Our R environment location
import time
import gc
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import rpy2.robjects as robjects
from rpy2.rinterface_lib import callbacks

# Suppress warnings
warnings.filterwarnings('ignore')

# Set R environment encoding for rpy2
callbacks._CCHAR_ENCODING = 'utf-8'

# Dataset paths
dataset_paths = {
    "gy_geno": "datasets/Wheat/gy_geno.csv",
    "gy_pheno": "datasets/Wheat/gy_pheno.csv"
}

# Phenotype dictionary mapping dataset_id_pheno to target columns
pheno_dict = {
    'gy_pheno': ['yield']
}

def check_data_quality(df, name):
    """Check for missing and negative values"""
    if df.isnull().any().any():
        print(f"Warning: Missing values in {name} data")
    if (df.select_dtypes(include=[np.number]) < 0).any().any():
        print(f"Warning: Negative values in {name} data")

def get_data(dataset_id):
    """Retrieve and preprocess data"""
    if 'corn' not in dataset_id:
        geno_data = pd.read_csv(dataset_paths[dataset_id + '_geno'],
                                # nrows=40, usecols=range(20)
                                )
        pheno_data = pd.read_csv(dataset_paths[dataset_id + '_pheno'], 
                                  # nrows=40
                                 )
        
        check_data_quality(geno_data, "Genotype")
        check_data_quality(pheno_data, "Phenotype")
        
        x_features = geno_data.columns.tolist()[1:]
        data = pd.merge(pheno_data, geno_data, left_on=pheno_data.columns[0], right_on=geno_data.columns[0])
    else:
        data = pd.read_csv(dataset_paths[dataset_id + '_pheno'])
        check_data_quality(data, "Integrated")
        x_features = data.columns.tolist()[1:-1]
        data = data.iloc[:, 1:]
    
    return data.dropna(), x_features

def measure_performance(y_true, y_pred_raw, start_time, memory_usage_mib, train_time, predict_time):
    """Calculate model performance metrics, handling NaN in predictions"""
    try:
        # Adjust prediction length to match true values
        if len(y_true) != len(y_pred_raw):
            min_len = min(len(y_true), len(y_pred_raw))
            y_true = y_true[:min_len]
            y_pred_raw = y_pred_raw[:min_len]

        # Filter out NaN predictions
        valid_mask = ~np.isnan(y_pred_raw)
        
        if not np.any(valid_mask):  # All predictions are NaN
            return {
                'Model': None,
                'Dataset': None,
                'Training Time (seconds)': train_time,
                'Prediction Time (seconds)': predict_time,
                'Max Memory Usage (MiB)': memory_usage_mib,
                'RMSE': np.nan, 'PCC': np.nan, 'MAE': np.nan, 'R2': np.nan,
                'PCC/RMSE': np.nan, 'Valid_predictions': 0, 'NaN_ratio': 1.0
            }

        y_true_clean = np.array(y_true)[valid_mask]
        y_pred_clean = np.array(y_pred_raw)[valid_mask]

        metrics = {
            'Model': None,
            'Dataset': None,
            'Training Time (seconds)': train_time,
            'Prediction Time (seconds)': predict_time,
            'Max Memory Usage (MiB)': memory_usage_mib,
            'RMSE': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            'PCC': pearsonr(y_true_clean, y_pred_clean)[0],
            'MAE': mean_absolute_error(y_true_clean, y_pred_clean),
            'R2': r2_score(y_true_clean, y_pred_clean),
            'Valid_predictions': np.sum(valid_mask)
        }
        metrics['PCC/RMSE'] = metrics['PCC'] / metrics['RMSE'] if metrics['RMSE'] != 0 else float('inf')
        metrics['NaN_ratio'] = 1 - (np.sum(valid_mask) / len(y_pred_raw))
        
        return metrics
    except Exception as e:
        print(f"Error in performance measurement: {e}")
        return {
            'Model': None,
            'Dataset': None,
            'Training Time (seconds)': train_time,
            'Prediction Time (seconds)': predict_time,
            'Max Memory Usage (MiB)': memory_usage_mib,
            'RMSE': np.nan, 'PCC': np.nan, 'MAE': np.nan, 'R2': np.nan,
            'PCC/RMSE': np.nan, 'Valid_predictions': 0, 'NaN_ratio': np.nan
        }

class RModelRunner:
    """Class to run R models"""
    def __init__(self):
        self.r = robjects.r

    @staticmethod
    def get_memory_usage_mib():
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)

    def fit_and_predict(self, method_name, X_train, y_train, X_test, save_dir):
        start_time = time.time()
        initial_memory = self.get_memory_usage_mib()
        
        try:
            X_train = np.asarray(X_train, dtype=np.float64).reshape(-1, 1) if X_train.ndim == 1 else np.asarray(X_train, dtype=np.float64)
            X_test = np.asarray(X_test, dtype=np.float64).reshape(-1, 1) if X_test.ndim == 1 else np.asarray(X_test, dtype=np.float64)
            y_train = np.asarray(y_train, dtype=np.float64)
            
            r_X_train = self.r['matrix'](robjects.FloatVector(X_train.flatten()), nrow=X_train.shape[0], ncol=X_train.shape[1], byrow=True)
            r_y_train = robjects.FloatVector(y_train)
            r_X_test = self.r['matrix'](robjects.FloatVector(X_test.flatten()), nrow=X_test.shape[0], ncol=X_test.shape[1], byrow=True)
            
            os.makedirs(save_dir, exist_ok=True)
            
            r_script_path = f'./R_utils/{method_name}.R'
            self.r['source'](r_script_path)
            r_result = self.r[method_name](r_X_train, r_y_train, r_X_test)
            
            # Extract training and prediction times
            train_time = r_result.rx2('train_time')[0] if 'train_time' in r_result.names else 0
            predict_time = r_result.rx2('predict_time')[0] if 'predict_time' in r_result.names else 0
            
            model_path = os.path.join(save_dir, f'{method_name}_model.rds')
            try:
                if isinstance(r_result, robjects.vectors.ListVector) and 'model' in r_result.names:
                    self.r('saveRDS')(r_result.rx2('model'), model_path)
            except Exception:
                pass
            
            predictions = None
            try:
                if isinstance(r_result, robjects.vectors.ListVector) and 'predictions' in r_result.names:
                    predictions = np.array(r_result.rx2('predictions'))
                    if predictions.ndim > 1:
                        predictions = predictions[:, 0]
                else:
                    predictions = np.array(r_result)
            except Exception:
                predictions = np.full(X_test.shape[0], np.nan)
            
            if np.sum(np.isnan(predictions)) > 0:
                print(f"Warning: NaN predictions for {method_name}.")
            
            memory_used_mib = self.get_memory_usage_mib() - initial_memory
            
            return predictions, train_time, predict_time, memory_used_mib
            
        except Exception as e:
            print(f"Error in {method_name}: {e}")
            return None, None, None, None

if __name__ == '__main__':
    print(f"Script start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    dataset_ids = [k.replace('_pheno', '') for k in dataset_paths.keys() if '_pheno' in k]
    
    models_to_evaluate = ['BayesB', 'GBLUP', 'RRBLUP']
    
    checkpoints_save_path = 'checkpoints'
    for dataset_id in dataset_ids:
        try:
            concat_data, x_features = get_data(dataset_id)
            
            for y_target_col in pheno_dict[dataset_id + '_pheno']:
                print(f"\nProcessing Dataset: {dataset_id}, Target: {y_target_col}")
                task_id = f'{dataset_id}_{y_target_col}'
                
                X = concat_data[x_features].values.astype(np.float32)
                y = concat_data[y_target_col].values.astype(np.float32)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                all_model_results_for_task = []
                
                for model_name in models_to_evaluate:
                    print(f"  Training Model: {model_name}...")
                    model_save_dir = os.path.join(checkpoints_save_path, task_id, model_name)
                    
                    r_model_runner = RModelRunner()
                    predictions, train_time, predict_time, memory_used_mib = r_model_runner.fit_and_predict(
                        model_name, X_train, y_train, X_test, model_save_dir
                    )
                    
                    if predictions is not None:
                        metrics = measure_performance(
                            y_test, predictions, time.time(), memory_used_mib, train_time, predict_time
                        )
                        metrics.update({
                            'Model': model_name,
                            'Dataset': task_id,
                        })
                        
                        # Reorder columns to match specified format
                        ordered_metrics = {
                            'Model': metrics['Model'],
                            'Dataset': metrics['Dataset'],
                            'Training Time (seconds)': metrics['Training Time (seconds)'],
                            'Prediction Time (seconds)': metrics['Prediction Time (seconds)'],
                            'Max Memory Usage (MiB)': metrics['Max Memory Usage (MiB)'],
                            'RMSE': metrics['RMSE'],
                            'PCC': metrics['PCC'],
                            'MAE': metrics['MAE'],
                            'R2': metrics['R2']
                        }
                        
                        all_model_results_for_task.append(ordered_metrics)
                        
                        # Save individual model metrics to XLSX
                        pd.DataFrame([ordered_metrics]).to_excel(
                            os.path.join(model_save_dir, f'{model_name}_metrics.xlsx'),
                            index=False
                        )
                        
                        # Save predictions
                        predictions_to_save = predictions[:len(y_test)] if len(predictions) >= len(y_test) else np.pad(predictions, (0, len(y_test) - len(predictions)), 'constant', constant_values=np.nan)
                        pd.DataFrame({
                            'true_values': y_test,
                            'predictions': predictions_to_save,
                            'is_nan': np.isnan(predictions_to_save)
                        }).to_excel(
                            os.path.join(model_save_dir, f'{model_name}_predictions.xlsx'),
                            index=False
                        )
                    gc.collect()
                
                if all_model_results_for_task:
                    results_df = pd.DataFrame(all_model_results_for_task)
                    # Ensure correct column order
                    results_df = results_df[['Model', 'Dataset', 'Training Time (seconds)', 'Prediction Time (seconds)',
                                           'Max Memory Usage (MiB)', 'RMSE', 'PCC', 'MAE', 'R2']]
                    aggregated_results_dir = os.path.join(checkpoints_save_path, task_id)
                    os.makedirs(aggregated_results_dir, exist_ok=True)
                    results_df.to_excel(os.path.join(aggregated_results_dir, 'R_performance.xlsx'), index=False)
                    print(f"  Aggregated results saved to: {os.path.join(aggregated_results_dir, 'R_performance.xlsx')}")
        
        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {e}")

    print(f"\nScript end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
