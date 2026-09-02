.libPaths("~/R/library")
library(dplyr)

results_dir <- if(dir.exists("Results/CORRECTED")) "Results/CORRECTED/" else if(dir.exists("../../Results/CORRECTED")) "../../Results/CORRECTED/" else "/Users/nazu.ds/Documents/Research Collections/Scientific_Reports_Submission_Package/For resubmission to another journal/Replication_Package/Results/CORRECTED/"

# BAS — aggregate selected variables from existing files
aggregate_bas_pips <- function(dataset) {
  files <- list.files(results_dir, pattern=paste0(dataset, "_.*_selected_variables.*\\.csv"), full.names=TRUE)
  files <- files[!grepl("results_BART|top_variables", files)]
  
  all_data <- list()
  for (f in files) {
    df <- read.csv(f)
    fname <- basename(f)
    
    method <- if(grepl("MICE", fname)) "MICE" else if(grepl("KNN", fname)) "KNN" else if(grepl("MEAN", fname)) "MEAN" else "missForest"
    mi <- if(grepl("wMI", fname)) "wMI" else "noMI"
    
    if (!("rank" %in% names(df))) {
      df <- df %>% group_by(fold) %>% mutate(rank = row_number()) %>% ungroup()
    }
    
    df$method <- method
    df$mi <- mi
    all_data[[length(all_data)+1]] <- df
  }
  
  combined <- bind_rows(all_data)
  has_pip <- "pip" %in% names(combined)
  combined %>%
    group_by(method, mi, variable) %>%
    summarise(
      mean_rank = mean(rank, na.rm=TRUE),
      mean_pip = if(has_pip) mean(pip, na.rm=TRUE) else NA_real_,
      freq = n(),
      .groups="drop"
    ) %>%
    arrange(method, mi, mean_rank)
}

# BART — aggregate selected variables from existing files
aggregate_bart_pips <- function(dataset) {
  pattern <- if(dataset == "MIMIC_REAL") "results_BART_MIMIC_REAL_.*_selected" else "results_BART_MI_REAL_.*_selected"
  files <- list.files(results_dir, pattern=pattern, full.names=TRUE)
  files <- files[!grepl("top_variables", files)]
  
  all_data <- list()
  for (f in files) {
    df <- read.csv(f)
    fname <- basename(f)
    
    method <- if(grepl("MICE", fname)) "MICE" else if(grepl("KNN", fname)) "KNN" else if(grepl("MEAN", fname)) "MEAN" else "missForest"
    mi <- if(grepl("wMI", fname)) "wMI" else "noMI"
    
    if (!("rank" %in% names(df))) {
      df <- df %>% group_by(fold) %>% mutate(rank = row_number()) %>% ungroup()
    }
    
    df$method <- method
    df$mi <- mi
    all_data[[length(all_data)+1]] <- df
  }
  
  bind_rows(all_data) %>%
    group_by(method, mi, variable) %>%
    summarise(mean_rank = mean(rank, na.rm=TRUE), freq = n(), .groups="drop") %>%
    arrange(method, mi, mean_rank)
}

# Generate top 10 per method and mi condition
for (dataset in c("MIMIC_REAL", "MI_REAL")) {
  cat("Processing BAS for", dataset, "\n")
  bas_agg <- aggregate_bas_pips(dataset)
  top_bas <- bas_agg %>% group_by(method, mi) %>% slice_min(mean_rank, n=10) %>% ungroup()
  write.csv(top_bas, file.path(results_dir, paste0(dataset, "_top_variables_BAS.csv")), row.names=FALSE)
  
  cat("Processing BART for", dataset, "\n")
  bart_agg <- aggregate_bart_pips(dataset)
  top_bart <- bart_agg %>% group_by(method, mi) %>% slice_min(mean_rank, n=10) %>% ungroup()
  write.csv(top_bart, file.path(results_dir, paste0(dataset, "_top_variables_BART.csv")), row.names=FALSE)
}

cat("Done.\n")
