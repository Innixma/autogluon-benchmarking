#!/usr/bin/env Rscript

# Plotting Kaggle Benchmark results: 

# To run this code, first specify these arguments appropriately: results_directory, raw_results_file, time_limit
# After specifying the arguments appropriately, you can run this script from command line via: 
# Rscript --vanilla autogluon_utils/benchmarking/evaluation/plotting/kagglebenchmarkplots.R data/results/ results_kaggle_wpercentile.csv 8h
# Note: order of arguments matters here.



library(grDevices)

run_from_cmdline = TRUE # running from bash
if (run_from_cmdline) {
  args = commandArgs(trailingOnly=TRUE)
  results_directory = args[1]
  results_file = args[2]
  time_limit = args[3]
} else { # Running from R
  # Example settings of these args within R:
  setwd("ANONYMOUS")
  results_directory = 'ANONYMOUS'  # directory where input/output files are located.
  results_file = 'results_kaggle_wpercentile.csv'  #  CSV File with performance numbers, must be located somewhere inside of resultsdir
  time_limit = '4h'  # Change this to compare different method runtimes (appended to end of names_tocompare)
  
}

# Working directory must be: setwd( autogluon-utils/ )
source("autogluon_utils/benchmarking/evaluation/plotting/plot_funcs.R")

print("Plotting Kaggle benchmark results with arguments:")
print(paste("results_directory:", results_directory))
print(paste("results_file:", results_file))
print(paste("time_limit:", time_limit))

# Names:
autogluon = "autogluon"
names_tocompare = c(autogluon, "autosklearn", "TPOT", "AutoWEKA", "H2OAutoML", "GoogleAutoMLTables")
official_names = c("AutoGluon", "auto-sklearn","TPOT   ","Auto-WEKA", "H2O AutoML", "GCP-Tables")
suffix = paste("_", time_limit, sep="")
input_subdirectory = "input/raw/"
output_subdirectory = "output/kaggle/"
input_dir = paste(results_directory, input_subdirectory, sep="")
output_dir = paste(results_directory, output_subdirectory, time_limit,"/", sep="")
autogluon_name = paste(autogluon,suffix,sep="")
alg_names = paste(names_tocompare,suffix,sep="")
# Column names for the results_file:
dataset_col = 'DATASET'
model_col = 'MODEL_NAME'
problem_type_col = 'PROBLEM_TYPE'
train_time_col = 'TIME_TRAIN_S'
# How to compare each baseline's performance vs autogluon's performance.
# Options include: "divide", "subtract", "absolute"


# First process dataframe of raw results (produced from python code evaluate_openml_results.py): 
df <- read.csv(paste(input_dir,results_file,sep=''))
print(head(df))
print(unique(df[[model_col]]))
print(c("Num datasets:", length(unique(df[[dataset_col]]))))

## Collect method-specific results:

performance_col = 'LEADER_RANK' # can also be 'METRIC_ERROR' (We assume lower values = better).
how_to_compare = "subtract" 

res = getPerfDF(df=df, performance_col=performance_col, 
          how_to_compare=how_to_compare, dataset_col=dataset_col, 
          model_col=model_col, problem_type_col=problem_type_col, 
          train_time_col=train_time_col, alg_names=alg_names, 
          problem_types=problem_types)
normdf = res[['normdf']]; problem_types = res[['problem_types']]
autogluon_performance = normdf[[autogluon_name]][1]

## Just plot performance:
plot_file = paste("KaggleLeaderboardPlot", suffix, ".pdf", sep="")
plotPerformance(normdf, timedf, names_tocompare, suffix, autogluon, how_to_compare=how_to_compare,
                official_names=official_names, autogluon_performance=autogluon_performance,
                cex_val=2, title="", 
                xlabel='Leaderboard Place (relative to AutoGluon)',
                problem_types=problem_types, show_legend=FALSE,
                savefile=paste(output_dir,plot_file, sep=''))

## Just plot legend:
legend_plot_file = paste("KaggleLegendPlot", suffix, ".pdf", sep="")
legend_text_width = rep(0.115, length(names_tocompare))
plotLegend(cex_val=2, names_tocompare=names_tocompare,
           autogluon=autogluon, official_names=official_names,
           legend_text_width=legend_text_width,
           savefile=paste(output_dir,legend_plot_file, sep=''))


## Repeat to plot Related Error-rate instead of leaderboard rank: 
how_to_compare = "divide"
performance_col = "METRIC_ERROR"
res = getPerfDF(df=df, performance_col=performance_col, 
                   how_to_compare=how_to_compare, dataset_col=dataset_col, 
                   model_col=model_col, problem_type_col=problem_type_col, 
                   train_time_col=train_time_col, alg_names=alg_names, 
                   problem_types=problem_types)
normdf = res[['normdf']]; problem_types = res[['problem_types']]
autogluon_performance = normdf[[autogluon_name]][1]
outlier_thres = 10
if (sum(normdf > outlier_thres, na.rm=TRUE) > 0) {t
  print("removing outlier values:")
  outs = unlist(as.list(normdf[normdf > outlier_thres]))
  print(outs[!is.na(outs)])
  normdf[normdf > outlier_thres] = NA
}

plot_file = paste("KaggleTestLossPlot", suffix, ".pdf", sep="")
plotPerformance(normdf, timedf, names_tocompare, suffix, autogluon, how_to_compare=how_to_compare,
                official_names=official_names, autogluon_performance=autogluon_performance,
                cex_val=2, title="", 
                xlabel='Loss on Test Data (relative to AutoGluon)',
                problem_types=problem_types, show_legend=FALSE,
                savefile=paste(output_dir,plot_file, sep=''))


## Repeat to just plot Absolute Percentiles:
how_to_compare="absolute"
res = getPerfDF(df=df, performance_col='LEADER_PERCENTILE', 
                 how_to_compare=how_to_compare, dataset_col=dataset_col, 
                 model_col=model_col, problem_type_col=problem_type_col, 
                 train_time_col=train_time_col, alg_names=alg_names, 
                 problem_types=problem_types)
normdf = res[['normdf']]; problem_types = res[['problem_types']]
# normdf = 1 - normdf # to get percentile rank


plot_file = paste("KaggleAbsolutePercentilePlot", suffix, ".pdf", sep="")
plotPercentiles(normdf, timedf, names_tocompare, suffix, autogluon, 
                how_to_compare=how_to_compare,
                official_names=official_names, autogluon_performance=autogluon_performance,
                cex_val=2, problem_types=problem_types, show_legend=FALSE,
                savefile=paste(output_dir,plot_file, sep=''))

## Repeat to plot absolute training times:
how_to_compare = "absolute"
res = getPerfDF(df=df, performance_col=train_time_col, 
                   how_to_compare=how_to_compare, dataset_col=dataset_col, 
                   model_col=model_col, problem_type_col=problem_type_col, 
                   train_time_col=train_time_col, alg_names=alg_names, 
                   problem_types=problem_types)
normdf = res[['normdf']]; problem_types = res[['problem_types']]
normdf = normdf / 60

plot_file = paste("KaggleTrainTimePlot", suffix, ".pdf", sep="")
plotPerformance(normdf, timedf, names_tocompare, suffix, autogluon, 
                how_to_compare=how_to_compare,
                official_names=official_names, autogluon_performance=autogluon_performance,
                cex_val=2, title="", 
                xlabel='Training Time (min)',
                problem_types=problem_types, log_axis=TRUE, show_legend=FALSE,
                savefile=paste(output_dir,plot_file, sep=''))




### Make this TRUE if you wish to also show runtimes next to performance ###
if (FALSE) {
timedf = data.frame(dataset = rownames(resdf))
for (algname in alg_names) {
  timedf[[algname]] = zero_vec
}
for (dat in as.character(timedf$dataset)) {
  index = which(timedf$dataset == dat)
  for (alg in alg_names) {
    og_index = which((df[[dataset_col]] == dat) & (df[[model_col]] == alg))
    if (length(og_index) > 0) {
      time = df[[train_time_col]][og_index]
    } else {
      warning(c("Missing index for: ", alg, " on dataset ", dat))
      time = NA
    }
    timedf[index,alg] = time
  }
}
rownames(timedf) = timedf$dataset
timedf$dataset = NULL
timedf = timedf / 60 # convert to minutes

# Plot performance and time:
plot_file = paste("KaggleLeaderBoardTimePlot", suffix, ".pdf", sep="")
plotPerformanceAndRuntime(normdf, timedf, names_tocompare, suffix, autogluon, 
    autogluon_performance=autogluon_performance, cex_val=2, title="", 
    xlabel='Leaderboard Place (relative to AutoGluon)',
    problem_types=problem_types, show_legend=TRUE,
    savefile=paste(output_dir,plot_file, sep=''))
}

