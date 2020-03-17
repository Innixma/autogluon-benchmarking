#!/usr/bin/env Rscript

# Plotting OpenML AutoML Benchmark results: 


# To run this code, first specify these arguments appropriately: results_directory, input_subdir, results_file, time_limit
# After specifying the arguments appropriately, you can run this script from command line via: 
# Rscript --vanilla autogluon_utils/benchmarking/evaluation/plotting/openmlbenchmarkplots.R data/results/ results_.csv 4h
# Note: order of arguments matters here.



library(grDevices)

run_from_cmdline = TRUE # running from bash
if (run_from_cmdline) {
  args = commandArgs(trailingOnly=TRUE)
  results_directory = args[1]
  input_subdirectory = args[2]
  results_file = args[3]
  time_limit = args[4]
} else { # Running from R
  # Example settings of these args within R:
  setwd("ANONYMOUS")
  results_directory = 'data/results/'  # directory where input/output files are located.
  input_subdirectory = 'output/openml/core/1h/'
  results_file = 'results_ranked_by_dataset_all.csv'   # CSV File with performance numbers, must be located somewhere inside of resultsdir
  time_limit = '1h'  # Change this to compare different method runtimes (appended to end of names_tocompare)
}

# Working directory must be: setwd( autogluon-utils/ )
source("autogluon_utils/benchmarking/evaluation/plotting/plot_funcs.R")

print("Plotting OpenML benchmark results with arguments:")
print(paste("results_directory:", results_directory))
print(paste("input_subdirectory:", input_subdirectory))
print(paste("results_file:", results_file))
print(paste("time_limit:", time_limit))

# Names:
output_subdir = "plots/"
autogluon = "autogluon"
names_tocompare = c(autogluon, "autosklearn", "TPOT", "AutoWEKA", "H2OAutoML", "GCPTables")
official_names = c("AutoGluon", "auto-sklearn","TPOT   ","Auto-WEKA", "H2O AutoML", "GCP-Tables")
suffix = paste("_", time_limit, sep="")

output_subdirectory = paste(results_directory, input_subdirectory, output_subdir, sep="")
input_dir = paste(results_directory, input_subdirectory, sep="")
output_dir = paste(results_directory, output_subdirectory, time_limit,"/", sep="")
autogluon_name = paste(autogluon,suffix,sep="")
alg_names = paste(names_tocompare,suffix,sep="")
# Column names for the results_file:
dataset_col = 'dataset'
model_col = 'framework'
problem_type_col = 'problem_type'
train_time_col = 'time_train_s'

# First process dataframe of raw results (produced from python code evaluate_openml_results.py): 
df <- read.csv(paste(input_dir,results_file,sep=''))
print(head(df))
print(unique(df[[model_col]]))
print(c("Num datasets:", length(unique(df[[dataset_col]]))))

# Create output_subdir folder to store plots:
dir.create(output_subdirectory)

## Collect method-specific results:

performance_col = 'metric_error'
how_to_compare = "divide" 
res = getPerfDF(df=df, performance_col=performance_col, 
                how_to_compare=how_to_compare, dataset_col=dataset_col, 
                model_col=model_col, problem_type_col=problem_type_col, 
                train_time_col=train_time_col, alg_names=alg_names, 
                problem_types=problem_types)
normdf = res[['normdf']]; problem_types = res[['problem_types']]
autogluon_performance = normdf[[autogluon_name]][1]
outlier_thres = 3000
if (sum(normdf > outlier_thres, na.rm=TRUE) > 0) {t
  print("removing outlier values:")
  outs = unlist(as.list(normdf[normdf > outlier_thres]))
  print(outs[!is.na(outs)])
  normdf[normdf > outlier_thres] = NA
}


## Just plot performance:
plotfilename = "OpenMLTestLossPlot"
plot_file = paste(plotfilename, suffix, ".pdf", sep="")
plotPerformance(normdf, timedf, names_tocompare, suffix, autogluon, how_to_compare=how_to_compare,
                official_names=official_names, autogluon_performance=autogluon_performance,
                cex_val=2, title="", 
                xlabel='Loss on Test Data (relative to AutoGluon)',
                problem_types=problem_types, show_legend=FALSE,
                savefile=paste(output_subdirectory,plot_file, sep=''))


## Just plot legend:
legend_plot_file = paste("LegendPlot", suffix, ".pdf", sep="")
legend_text_width = rep(0.115, length(names_tocompare))
plotLegend(cex_val=2, names_tocompare=names_tocompare,
           autogluon=autogluon, official_names=official_names,
           legend_text_width=legend_text_width,
           savefile=paste(output_subdirectory,legend_plot_file, sep=''))


## Repeat to plot absolute training times:
how_to_compare = "absolute"
res = getPerfDF(df=df, performance_col=train_time_col, 
                how_to_compare=how_to_compare, dataset_col=dataset_col, 
                model_col=model_col, problem_type_col=problem_type_col, 
                train_time_col=train_time_col, alg_names=alg_names, 
                problem_types=problem_types)
normdf = res[['normdf']]; problem_types = res[['problem_types']]
normdf = normdf / 60

plot_file = paste("OpenMLTrainTimePlot", suffix, ".pdf", sep="")
plotPerformance(normdf, timedf, names_tocompare, suffix, autogluon, 
                how_to_compare=how_to_compare,
                official_names=official_names, autogluon_performance=autogluon_performance,
                cex_val=2, title="", 
                xlabel='Training Time (min)',
                problem_types=problem_types, log_axis=TRUE, show_legend=FALSE,
                savefile=paste(output_subdirectory,plot_file, sep=''))
