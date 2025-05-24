RRBLUP <- function(X_train, y_train, X_test) {
  # Load or install rrBLUP package
  if (!require("rrBLUP", character.only = TRUE)) {
    install.packages("rrBLUP", repos = "https://cloud.r-project.org")
  }
  library(rrBLUP)
  
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
  
  # Create training data frame
  training_data <- data.frame(y = y_train, id = rownames(X_train))
  
  # Train model using rrBLUP
  rrblup_fit <- mixed.solve(y = training_data$y, Z = X_train)
  
  # Calculate training duration
  train_duration <- as.numeric(difftime(Sys.time(), train_start_time, units = "secs"))
  
  # Record prediction start time
  predict_start_time <- Sys.time()
  
  # Predict on test set
  predictions <- X_test %*% rrblup_fit$u
  
  # Calculate prediction duration
  predict_duration <- as.numeric(difftime(Sys.time(), predict_start_time, units = "secs"))
  
  # Process prediction results
  if (is.list(predictions) && "predictions" %in% names(predictions)) {
    final_predictions <- predictions$predictions
  } else {
    final_predictions <- predictions
  }
  
  # Return model, predictions, and timing metrics
  result <- list(
    model = rrblup_fit,
    predictions = final_predictions,
    train_time = train_duration,
    predict_time = predict_duration
  )
  return(result)
}
