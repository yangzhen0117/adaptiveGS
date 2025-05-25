BayesB <- function(X_train, y_train, X_test) {
  # This function implements the BayesB genomic prediction method.
  # It relies on the 'BWGS' R package.
  # Package Source: https://cran.r-project.org/web/packages/BWGS/index.html
  # Charmet G, Tran LG, Auzanneau J, Rincent R, Bouchet S. BWGS: A R package for genomic selection and its application to a wheat breeding programme. PLoS One. 2020 Apr 23;15(4):e0232422. doi: 10.1371/journal.pone.0232422. PMID: 32240182; PMCID: PMC7141418.
  
  if (!require("BWGS", character.only = TRUE)) {
    install.packages("BWGS", repos = "https://cloud.r-project.org")
  }
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
  
  result <- bwgs.predict(geno_train = X_train,
                         pheno_train = y_train,
                         geno_target = X_test,
                         FIXED_train = "NULL", FIXED_target = "NULL", MAXNA = 0.2, MAF = 0.05, geno.reduct.method = "NULL", reduct.size = "NULL", r2 = "NULL", pval = "NULL", MAP = "NULL", geno.impute.method = "NULL", predict.method = "BB")
  
  # Calculate training duration (seconds)
  train_duration <- as.numeric(difftime(Sys.time(), train_start_time, units = "secs"))
  
  # Process results
  if (is.list(result) && "predictions" %in% names(result)) {
    predictions <- result$predictions
    model <- result
  } else {
    predictions <- result
    model <- list(predictions = result)
  }
  
  predict_start_time <- Sys.time()
  
  # Calculate prediction duration 
  predict_duration <- as.numeric(difftime(Sys.time(), predict_start_time, units = "secs"))
  
  # Return results with timing metrics
  result <- list(
    model = model,
    predictions = predictions,
    train_time = train_duration,
    predict_time = predict_duration
  )
  return(result)
}