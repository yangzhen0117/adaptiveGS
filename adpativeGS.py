import numpy as np
import pandas as pd
import time
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet,LinearRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,StackingRegressor
from scipy.stats import pearsonr, uniform, randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error,mean_absolute_error, r2_score
from memory_profiler import memory_usage

# GY
snp_df = pd.read_excel('gy_geno.xlsx')
snp_df = snp_df.iloc[:,1:]
pheno_df = pd.read_excel('gy_pheno.xlsx')
pheno_df =pheno_df.iloc[:,1]

X = snp_df
y = pheno_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define scorers
def pcc_scorer(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def measure_performance(model, model_name):
    start_time = time.time()
    mem_usage = memory_usage((model.fit, (X_train, y_train)), max_usage=True)
    end_time = time.time()

    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    pcc = pearsonr(y_test, pred)[0]
    pcc_rmse = pcc / rmse
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    execution_time = end_time - start_time

    # Ensure mem_usage is a list before calling max
    if isinstance(mem_usage, float):
        max_memory_usage = mem_usage
    else:
        max_memory_usage = max(mem_usage)

    return {
        'Model': model_name,
        'Execution Time (seconds)': execution_time,
        'Max Memory Usage (MiB)': max_memory_usage,
        'RMSE': rmse,
        'PCC': pcc,
        'PCC/RMSE': pcc_rmse,
        'MAE': mae,
        'R2': r2
    }

# Define all models
svr = SVR()
krr = KernelRidge(alpha=0.1, kernel='rbf')
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
model_lgb = lgb.LGBMRegressor()
model_xgb = xgb.XGBRegressor()
model_rf = RandomForestRegressor(n_estimators=500, random_state=42)
model_gbdt = GradientBoostingRegressor()

# Get model name and object mapping
models = {
    'SVR': svr,
    'Kernel Ridge': krr,
    'ElasticNet': enet,
    'LightGBM': model_lgb,
    'XGBoost': model_xgb,
    'RandomForest': model_rf,
    'GradientBoosting': model_gbdt
}

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    results = []
    for model_name, model in models.items():
        result = measure_performance(model, model_name)
        results.append(result)

    # Convert the result to a DataFrame
    results_df = pd.DataFrame(results)

    # Selection of first three models based on 'PCC/RMSE'
    top3_models = results_df.nlargest(3, 'PCC/RMSE')

    # Automatically generate a list of estimators
    selected_estimators = [(row['Model'], models[row['Model']]) for _, row in top3_models.iterrows()]

    # The first three models selected for output and their names
    print(f"Selected top 3 models for Stacking: {selected_estimators}")

    # StackingRegressor using the selected model
    stack5 = StackingRegressor(
        estimators=selected_estimators,
        final_estimator=LinearRegression()
    )

    # Dynamic creation of parameterized search spaces
    param_distributions = {}
    if 'SVR' in top3_models['Model'].values:
        param_distributions.update({
            'SVR__C': uniform(0.1, 1000),
            'SVR__gamma': uniform(0.0001, 1),
            'SVR__kernel': ['linear', 'poly', 'rbf'],
        })
    if 'Kernel Ridge' in top3_models['Model'].values:
        param_distributions.update({
            'Kernel Ridge__alpha': uniform(0.01, 10),
            'Kernel Ridge__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        })
    if 'ElasticNet' in top3_models['Model'].values:
        param_distributions.update({
            'ElasticNet__alpha': uniform(0.01, 1),
            'ElasticNet__l1_ratio': uniform(0.01, 1),
        })
    if 'RandomForest' in top3_models['Model'].values:
        param_distributions.update({
            'RandomForest__n_estimators': randint(50, 1000),
            'RandomForest__max_depth': randint(3, 20),
            'RandomForest__min_samples_split': randint(2, 20),
            'RandomForest__min_samples_leaf': randint(1, 10),
        })
    if 'LightGBM' in top3_models['Model'].values:
        param_distributions.update({
            'LightGBM__num_leaves': randint(20, 150),
            'LightGBM__max_depth': randint(3, 20),
            'LightGBM__learning_rate': uniform(0.01, 0.3),
            'LightGBM__n_estimators': randint(50, 1000),
        })
    if 'XGBoost' in top3_models['Model'].values:
        param_distributions.update({
            'XGBoost__max_depth': randint(3, 20),
            'XGBoost__learning_rate': uniform(0.01, 0.3),
            'XGBoost__n_estimators': randint(50, 1000),
            'XGBoost__gamma': uniform(0.01, 1),
            'XGBoost__subsample': uniform(0.5, 1),
            'XGBoost__colsample_bytree': uniform(0.5, 1),
        })
    if 'GradientBoosting' in top3_models['Model'].values:
        param_distributions.update({
            'GradientBoosting__learning_rate': uniform(0.01, 0.3),
            'GradientBoosting__n_estimators': randint(50, 1000),
            'GradientBoosting__max_depth': randint(3, 20),
            'GradientBoosting__min_samples_split': randint(2, 20),
            'GradientBoosting__min_samples_leaf': randint(1, 10),
        })
    # Randomized Search Hyperparameters
    random_search = RandomizedSearchCV(
        estimator=stack5,
        param_distributions=param_distributions,
        n_iter=10,
        cv=KFold(n_splits=10, shuffle=True, random_state=42),
        scoring=make_scorer(pcc_scorer, greater_is_better=True),
        n_jobs=-1,
        random_state=42
    )

    # Fitting Stacking Models and Measuring Performance
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    stacking_result = measure_performance(best_model, 'Stacking')

    # Add Stacking model results to the results list
    results.append(stacking_result)
    # Save results as an Excel file
    results_df = pd.DataFrame(results)
    results_df.to_excel('model_results.xlsx', index=False)
    print("Results saved to model_results.xlsx")


