GBLUP <- function(X_train, y_train, X_test) {
  library(BWGS)
  set.seed(42)
  
  # Record training start time
  train_start_time <- Sys.time()
  
  # Convert data to matrices
  X_train <- as.matrix(X_train)
  rownames(X_train) <- paste("name", 1:nrow(X_train), sep = "") 
  colnames(X_train) <- paste("col", 1:ncol(X_train), sep = "") 

  y_train <- unlist(y_train)
  names(y_train) <- paste("name", 1:length(y_train), sep = "")
  
  X_test <- as.matrix(X_test)
  rownames(X_test) <- paste("nme", 1:nrow(X_test), sep = "") 
  colnames(X_test) <- paste("col", 1:ncol(X_test), sep = "") 
  
  # Execute model training and prediction
  result <- bwgs.predict(geno_train = X_train,
                         pheno_train = y_train,
                         geno_target = X_test,
                         FIXED_train = "NULL", FIXED_target = "NULL", MAXNA = 0.1, MAF = 0.05, geno.reduct.method = "NULL", reduct.size = "NULL", r2 = "NULL", pval = "NULL", MAP = "NULL", geno.impute.method = "NULL", predict.method = "GBLUP")
  
  
  # Calculate training duration (seconds)
  train_duration <- as.numeric(difftime(Sys.time(), train_start_time, units = "secs"))
  
  # Process results structure
  if (is.list(result) && "predictions" %in% names(result)) {
    predictions <- result$predictions
    model <- result
  } else {
    predictions <- result
    model <- list(predictions = result)  
  }
  
  # Record prediction start time (if prediction is separate step)
  predict_start_time <- Sys.time()
  
  # Calculate prediction duration (assuming prediction is integrated in bwgs.predict)
  predict_duration <- as.numeric(difftime(Sys.time(), predict_start_time, units = "secs"))
  
  # Return results with timing metrics
  result <- list(
    model = model,
    predictions = predictions,
    train_time = train_duration,  # Training time in seconds
    predict_time = predict_duration  # Prediction time in seconds
  )
  return(result)
}