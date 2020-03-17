#!/usr/bin/env Rscript

# Helper methods for plotting benchmark results

KAGGLE_ABBREVS = list(
  'house-prices',
  'mercedes-benz',
  'santander-value',
  'allstate-claims',
  'bnp-paribas',
  'santander-transaction',
  'santander-satisfaction',
  'porto-seguro',
  'walmart-recruiting',
  'ieee-fraud',
  'otto-group'
)
names(KAGGLE_ABBREVS) = c(
  'house-prices-advanced-regression-techniques',
  'mercedes-benz-greener-manufacturing',
  'santander-value-prediction-challenge',
  'allstate-claims-severity',
  'bnp-paribas-cardif-claims-management',
  'santander-customer-transaction-prediction',
  'santander-customer-satisfaction',
  'porto-seguro-safe-driver-prediction',
  'walmart-recruiting-trip-type-classification',
  'ieee-fraud-detection',
  'otto-group-product-classification-challenge'
)

pch_autogluon = 20 # symbol used for autogluon point in plot
pch_val = 18 # symbol for other points
possible_colors <- c('blue2', 'firebrick2', 'darkolivegreen4', 'darkgoldenrod', 'darkorchid', 'cyan3', 'gray0', 'greenyellow')


plotPerformance <- function(results_df, time_df, names_tocompare, suffix, autogluon, how_to_compare = 'divide',
                            official_names = NULL, autogluon_performance = 1.0, problem_types=NULL, 
                            title="", xlabel='Loss (relative to AutoGluon)', 
                            alpha=0.6, cex_val = 2, log_axis = NULL, show_legend = TRUE, savefile=NULL,
                            width = NULL, height = NULL) {
  if (is.null(log_axis)) {
    log_axis = (how_to_compare == "divide") # Only log-axis when plotting performance ratios
  }
  point_cex = 2.2*cex_val # size of points
  label_cex = 1.25*cex_val # size of axis names
  legend_cex = label_cex
  grid_cex = cex_val
  yaxis_cex = 1.1*cex_val # unique for single plot
  axislabel_line = 4
  if (is.null(problem_types)) {
    problem_types = rep("binary", nrow(results_df))
  }
  if (is.null(official_names)) {
    official_names = names_tocompare
  }
  keep_inds = names_tocompare != autogluon
  names_tocompare = names_tocompare[keep_inds] # remove if it is part of this list
  official_names = official_names[keep_inds]
  method_colors = rep("",length(names_tocompare))
  pch_vals = rep(pch_val, length(method_colors))
  for (i in 1:length(names_tocompare)) {
    method_colors[i] = adjustcolor(possible_colors[i], alpha = alpha)
  }
  if ((how_to_compare != 'subtract') && (how_to_compare != 'divide')) {
    names_tocompare = c(names_tocompare, autogluon) # will be added back to official_names later
    method_colors = c(method_colors, adjustcolor("black", alpha=alpha))
    official_names = c(official_names, "AutoGluon")
    pch_vals = c(pch_vals, pch_autogluon)
  }
  alg_names = paste(names_tocompare, suffix, sep="")
  names_tocompare[names_tocompare == 'H2OAutoML'] = 'H2O'
  # results_df = replaceOutliers(results_df)
  # Create plots:
  if (!is.null(savefile)) {
    filestr = tolower(savefile)
    if (substr(filestr, nchar(filestr)-3+1, nchar(filestr)) == 'png') {
      png(savefile, width = 1536, height = 1024)
    } else if (substr(filestr, nchar(filestr)-3+1, nchar(filestr)) == 'pdf') {
      if ((!is.null(width)) && (!is.null(height))) {
        pdf(savefile, width = width, height = height)
      } else {
        pdf(savefile, width = 16, height = 12)
      }
    } else {
      stop(paste("File-extension in savefile must be either png or pdf, not: ", substr(filestr, nchar(filestr)-3+1, nchar(filestr))))
    }
  }
  # Plot performance results:
  y_vals = nrow(results_df):1
  worst_perf = min(apply(results_df, 2, function(x) min(x, na.rm = TRUE)))
  best_perf = max(apply(results_df, 2, function(x) max(x, na.rm = TRUE)))
  x_lim = c(0.99*worst_perf, 1.01*best_perf)
  log_str = ""
  if (log_axis) {
    log_str = "x"
  }
  par(las=1, mar=c(7.0, 18.0, 0.5, 0.5))
  if (log_axis && (min(c(worst_perf,best_perf))==0)) { # cannot log-scale 0s
    results_df = as.data.frame(apply(results_df, c(1,2), function(x) (x+1)))
    new_max_x = 1.01*max(apply(results_df, 2, function(x) max(x, na.rm = TRUE)))
    x_lim = c(1, new_max_x)
    autogluon_performance = autogluon_performance + 1
    plot(results_df[[1]], y_vals, xlim=x_lim, type='n', yaxt='n', xaxt='n', log=log_str, 
         main="", ylab='', xlab='', cex=cex_val, cex.axis=cex_val, cex.lab = label_cex)
    xticklocs = c(0.1, 0.5, 1, 2, 5, 10, 20, 50) + 1 
    axis(side=1,at=xticklocs, labels = xticklocs-1, cex.axis = cex_val)
  } else {
    plot(results_df[[1]], y_vals, xlim=x_lim, type='n', yaxt='n', log=log_str, 
       main="", ylab='', xlab='', cex=cex_val, cex.axis=cex_val, cex.lab = label_cex)
  }
  title(xlab = xlabel, cex.lab = label_cex, line = axislabel_line)
  dataset_names = getDatasetNames(results_df)
  axis(side=2,at=y_vals[problem_types == 'binary'],labels = dataset_names[problem_types == 'binary'], cex.axis = yaxis_cex)
  if ('multiclass' %in% problem_types) {
    axis(side=2,at=y_vals[problem_types == 'multiclass'], labels=dataset_names[problem_types == 'multiclass'], 
         col.axis='darkorchid4', cex.axis=yaxis_cex)
  }
  if ('regression' %in% problem_types) {
    axis(side=2,at=y_vals[problem_types == 'regression'], labels=dataset_names[problem_types == 'regression'],
         col.axis = 'darkorange', cex.axis=yaxis_cex)
  }
  ngridpt = ceiling(length(y_vals)/2)
  grid_col = 'snow4'
  grid(ny=0, nx=ngridpt, lwd = grid_cex, col=grid_col)
  for (i in 1:length(y_vals)) {
    abline(h=y_vals[i],lwd = grid_cex,col=grid_col, lty="dotted")
  }
  for (i in 1:length(names_tocompare)) {
    x_vals = results_df[[alg_names[i]]]
    points(x_vals[!is.na(x_vals)], y_vals[!is.na(x_vals)], col = method_colors[i], pch = pch_vals[i], cex = point_cex)
  }
  if (names_tocompare[length(names_tocompare)] != autogluon) {
    abline(v=autogluon_performance, lwd=point_cex*1.25) # Autogluon performance on top
  }
  # Plot legend:
  if (show_legend) {
    par(mai=c(0,0,0,0))
    plot.new()
    if (names_tocompare[length(names_tocompare)] != autogluon) {
      official_names = c(official_names, "AutoGluon")
      method_colors = c(method_colors, adjustcolor("black", alpha=alpha))
      pch_vals = c(pch_vals, pch_autogluon)
    }
    legend(x="center", ncol=length(names_tocompare), legend = official_names, col = method_colors, pch=pch_vals, text.col = method_colors, cex = legend_cex)
  }
  if (!is.null(savefile)) {
    print(c("Plot saved in:", savefile))
    dev.off()
  }
}


plotLegend <- function(names_tocompare, autogluon, official_names=NULL, 
                       alpha=0.8, cex_val = 2, savefile=NULL, legend_text_width=NULL) {
  if (is.null(official_names)) {
    official_names = names_tocompare
  }
  keep_inds = names_tocompare != autogluon
  names_tocompare = names_tocompare[keep_inds] # remove if it is part of this list
  official_names = official_names[keep_inds]
  label_cex = 1.25*cex_val # size of axis names
  legend_cex = label_cex
  pch_vals = rep(18, length(names_tocompare))
  method_colors = rep("",length(names_tocompare))
  for (i in 1:length(names_tocompare)) {
    method_colors[i] = adjustcolor(possible_colors[i], alpha = alpha)
  }
  # Create plots:
  if (!is.null(savefile)) {
    filestr = tolower(savefile)
    if (substr(filestr, nchar(filestr)-3+1, nchar(filestr)) == 'png') {
      png(savefile, width = 2048, height = 1024)
    } else if (substr(filestr, nchar(filestr)-3+1, nchar(filestr)) == 'pdf') {
      pdf(savefile, width = 24, height = 12)
    } else {
      stop(paste("File-extension in savefile must be either png or pdf, not: ", substr(filestr, nchar(filestr)-3+1, nchar(filestr))))
    }
  }
  plot.new()
  official_names = c(official_names, "AutoGluon")
  method_colors = c(method_colors, adjustcolor("black", alpha=alpha))
  pch_vals = c(pch_vals, pch_autogluon)
  if (is.null(legend_text_width)) {
    legend(x="center", legend = official_names, 
         col = method_colors, pch=pch_vals, text.col = method_colors, 
         cex = legend_cex, horiz=TRUE, bty = "n")
  } else {
    legend(x="center", legend = official_names, 
           col = method_colors, pch=pch_vals, text.col = method_colors, 
           cex=legend_cex, horiz=TRUE,bty="n", text.width=legend_text_width)
  }
  if (!is.null(savefile)) {
    print(c("Plot saved in:", savefile))
    dev.off()
  }
}

# Special plot for absolute percentiles
plotPercentiles <- function(results_df, time_df, names_tocompare, suffix, autogluon, how_to_compare = 'divide',
                            official_names = NULL, autogluon_performance = 1.0, problem_types=NULL, 
                            title="", xlabel='Percentile Rank on Leaderboard', 
                            alpha=0.6, cex_val = 2, show_legend = TRUE, savefile=NULL,
                            width = NULL, height = NULL) {
  point_cex = 2.2*cex_val # size of points
  label_cex = 1.25*cex_val # size of axis names
  legend_cex = label_cex
  grid_cex = cex_val
  yaxis_cex = 1.1*cex_val # unique for single plot
  axislabel_line = 4
  if (is.null(problem_types)) {
    problem_types = rep("binary", nrow(results_df))
  }
  if (is.null(official_names)) {
    official_names = names_tocompare
  }
  keep_inds = names_tocompare != autogluon
  names_tocompare = names_tocompare[keep_inds] # remove if it is part of this list
  official_names = official_names[keep_inds]
  method_colors = rep("",length(names_tocompare))
  pch_vals = rep(pch_val, length(method_colors))
  for (i in 1:length(names_tocompare)) {
    method_colors[i] = adjustcolor(possible_colors[i], alpha = alpha)
  }
  if ((how_to_compare != 'subtract') && (how_to_compare != 'divide')) {
    names_tocompare = c(names_tocompare, autogluon) # will be added back to official_names later
    method_colors = c(method_colors, adjustcolor("black", alpha=alpha))
    official_names = c(official_names, "AutoGluon")
    pch_vals = c(pch_vals, pch_autogluon)
  }
  alg_names = paste(names_tocompare, suffix, sep="")
  names_tocompare[names_tocompare == 'H2OAutoML'] = 'H2O'
  
  # Create plots:
  if (!is.null(savefile)) {
    filestr = tolower(savefile)
    if (substr(filestr, nchar(filestr)-3+1, nchar(filestr)) == 'png') {
      png(savefile, width = 1536, height = 1024)
    } else if (substr(filestr, nchar(filestr)-3+1, nchar(filestr)) == 'pdf') {
      if ((!is.null(width)) && (!is.null(height))) {
        pdf(savefile, width = width, height = height)
      } else {
        pdf(savefile, width = 16, height = 12)
      }
    } else {
      stop(paste("File-extension in savefile must be either png or pdf, not: ", substr(filestr, nchar(filestr)-3+1, nchar(filestr))))
    }
  }
  # Plot performance results:
  y_vals = nrow(results_df):1
  worst_perf = min(apply(results_df, 2, function(x) min(x, na.rm = TRUE)))
  best_perf = max(apply(results_df, 2, function(x) max(x, na.rm = TRUE)))
  x_lim = c(0, 1) # just for percentiles
  par(las=1, mar=c(7.0, 18.0, 0.5, 0.5))
  plot(results_df[[1]], y_vals, xlim=x_lim, type='n', yaxt='n', xaxt = 'n', # special x-axis just for percentiles plot
       main="", ylab='', xlab='', cex=cex_val, cex.axis=cex_val, cex.lab = label_cex)
  title(xlab = xlabel, cex.lab = label_cex, line = axislabel_line)
  dataset_names = getDatasetNames(results_df)
  scaleFUN <- function(x) sprintf("%.1f", x)
  xticks = seq(0,1,0.1)
  axis(side=1,at=xticks,labels = scaleFUN(rev(xticks)), cex.axis = yaxis_cex) # specially just for percentiles
  axis(side=2,at=y_vals[problem_types == 'binary'],labels = dataset_names[problem_types == 'binary'], cex.axis = yaxis_cex)
  if ('multiclass' %in% problem_types) {
    axis(side=2,at=y_vals[problem_types == 'multiclass'], labels=dataset_names[problem_types == 'multiclass'], 
         col.axis='darkorchid4', cex.axis=yaxis_cex)
  }
  if ('regression' %in% problem_types) {
    axis(side=2,at=y_vals[problem_types == 'regression'], labels=dataset_names[problem_types == 'regression'],
         col.axis = 'darkorange', cex.axis=yaxis_cex)
  }
  ngridpt = ceiling(length(y_vals)/2)
  grid_col = 'snow4'
  grid(ny=0, nx=ngridpt, lwd = grid_cex, col=grid_col)
  for (i in 1:length(y_vals)) {
    abline(h=y_vals[i],lwd = grid_cex,col=grid_col, lty="dotted")
  }
  for (i in 1:length(names_tocompare)) {
    x_vals = results_df[[alg_names[i]]]
    points(x_vals[!is.na(x_vals)], y_vals[!is.na(x_vals)], col = method_colors[i], pch = pch_vals[i], cex = point_cex)
  }
  if (names_tocompare[length(names_tocompare)] != autogluon) {
    abline(v=autogluon_performance, lwd=point_cex*1.25) # Autogluon performance on top
  }
  # Plot legend:
  if (show_legend) {
    par(mai=c(0,0,0,0))
    plot.new()
    if (names_tocompare[length(names_tocompare)] != autogluon) {
      official_names = c(official_names, "AutoGluon")
      method_colors = c(method_colors, adjustcolor("black", alpha=alpha))
      pch_vals = c(pch_vals, pch_autogluon)
    }
    legend(x="center", ncol=length(names_tocompare), legend = official_names, col = method_colors, pch=pch_vals, text.col = method_colors, cex = legend_cex)
  }
  if (!is.null(savefile)) {
    print(c("Plot saved in:", savefile))
    dev.off()
  }
}


plotLegend <- function(names_tocompare, autogluon, official_names=NULL, 
                       alpha=0.8, cex_val = 2, savefile=NULL, legend_text_width=NULL) {
  if (is.null(official_names)) {
    official_names = names_tocompare
  }
  keep_inds = names_tocompare != autogluon
  names_tocompare = names_tocompare[keep_inds] # remove if it is part of this list
  official_names = official_names[keep_inds]
  label_cex = 1.25*cex_val # size of axis names
  legend_cex = label_cex
  pch_vals = rep(18, length(names_tocompare))
  method_colors = rep("",length(names_tocompare))
  for (i in 1:length(names_tocompare)) {
    method_colors[i] = adjustcolor(possible_colors[i], alpha = alpha)
  }
  # Create plots:
  if (!is.null(savefile)) {
    filestr = tolower(savefile)
    if (substr(filestr, nchar(filestr)-3+1, nchar(filestr)) == 'png') {
      png(savefile, width = 2048, height = 1024)
    } else if (substr(filestr, nchar(filestr)-3+1, nchar(filestr)) == 'pdf') {
      pdf(savefile, width = 24, height = 12)
    } else {
      stop(paste("File-extension in savefile must be either png or pdf, not: ", substr(filestr, nchar(filestr)-3+1, nchar(filestr))))
    }
  }
  plot.new()
  official_names = c(official_names, "AutoGluon")
  method_colors = c(method_colors, adjustcolor("black", alpha=alpha))
  pch_vals = c(pch_vals, pch_autogluon)
  if (is.null(legend_text_width)) {
    legend(x="center", legend = official_names, 
           col = method_colors, pch=pch_vals, text.col = method_colors, 
           cex = legend_cex, horiz=TRUE, bty = "n")
  } else {
    legend(x="center", legend = official_names, 
           col = method_colors, pch=pch_vals, text.col = method_colors, 
           cex=legend_cex, horiz=TRUE,bty="n", text.width=legend_text_width)
  }
  if (!is.null(savefile)) {
    print(c("Plot saved in:", savefile))
    dev.off()
  }
}

getDatasetNames <- function(results_df) {
  dataset_names = rownames(results_df)
  for (i in 1:length(KAGGLE_ABBREVS)) {
    if (dataset_names[i] %in% names(KAGGLE_ABBREVS)) {
      dataset_names[i] = KAGGLE_ABBREVS[[dataset_names[i]]]
    }
  }
  dataset_names = shortStr(dataset_names, maxlen=17) # maxlen 17 is the same truncation value used in OpenML benchmark results plots
  return(dataset_names)
}

shortStr <- function(strings, maxlen=17) {
  shortened = strings
  for (i in 1:length(strings)) {
    if (nchar(strings[i]) > maxlen) {
      shortened[i] = paste(substr(strings[i],start = 1, stop=maxlen-2),'...',sep="")
    }
  }
  return(tolower(shortened))
}

replaceOutliers <- function(df) {
  for (j in 1:ncol(df)) {
    vals = df[,j]
    quantile80 = quantile(vals, probs=0.8,na.rm=TRUE)
    out_inds = vals > 20*quantile80
    if (sum(out_inds, na.rm=TRUE) > 1) {
      stop("should not remove more than one point as outlier")
    } else if (sum(out_inds, na.rm=TRUE) == 1) {
      print("removed outlier")
    }
    vals[out_inds] = NA
    df[,j] = vals
  }
  return(df)
}


getPerfDF <- function(df, performance_col, how_to_compare, problem_types,
                      dataset_col, model_col,problem_type_col,
                      train_time_col, alg_names) {
  
  return_obj = list()
  zero_vec = rep(0, length(unique(df[[dataset_col]])))
  resdf = data.frame(dataset = unique(df[[dataset_col]]))
  for (algname in alg_names) {
    resdf[[algname]] = zero_vec
  }
  
  problem_types = rep("", nrow(resdf))
  for (i in 1:nrow(resdf)) {
    index = which(df[[dataset_col]] == resdf$dataset[i])[1]
    problem_types[i] = as.character(df[[problem_type_col]][index])
  }
  
  ord = order(problem_types)
  problem_types = problem_types[ord]
  resdf = resdf[ord,]
  return_obj[['problem_types']] = problem_types
  
  for (dat in as.character(resdf$dataset)) {
    index = which(resdf$dataset == dat)
    for (alg in alg_names) {
      og_index = which((df[[dataset_col]] == dat) & (df[[model_col]] == alg))
      if (length(og_index) > 0) {
        err = df[[performance_col]][og_index]
        # type = df$metric_type[og_index]
      } else {
        warning(c("Missing index for: ", alg, " on dataset ", dat))
        err = NA
        # type = NA
      }
      # resdf[index,alg] = err2perf(err, type)
      resdf[index,alg] = err
    }
  }  
  rownames(resdf) = resdf$dataset
  resdf$dataset = NULL
  normdf = resdf
  for (col in 1:ncol(resdf)) {
    if (how_to_compare == 'subtract') {
      normalized_val = resdf[[col]] - resdf[[autogluon_name]]
    } else if (how_to_compare == 'divide') {
      normalized_val = resdf[[col]] / resdf[[autogluon_name]]
      if (min(resdf[[autogluon_name]], na.rm=TRUE) == 0) {
        zero_inds = which(resdf[[autogluon_name]] == 0)
        zero_inds = zero_inds[!is.na(zero_inds)]
        for (zero_ind in zero_inds) {
          if (!is.na(resdf[[col]][zero_ind])) {
            if (resdf[[col]][zero_ind] == 0) {
              normalized_val[zero_ind] = 1
            } else {
              normalized_val[zero_ind] = (resdf[[col]][zero_ind] + 1)/(0 + 1) # add one to each
            }
          }
        }
      }
    } else { # absolute
      normalized_val = resdf[[col]] # no relative rescaling
    }
    normdf[[col]] = normalized_val
  }
  return_obj[['normdf']] = normdf
  return(return_obj)
}



## Older plotting function:
plotPerformanceAndRuntime <- function(results_df, time_df, names_tocompare, 
                                      suffix, autogluon, official_names=NULL, autogluon_performance = 1.0, 
                                      title="", xlabel='Loss (relative to AutoGluon)', problem_types=NULL, 
                                      alpha=0.6, cex_val = 2, log_axis = FALSE, show_legend = TRUE, savefile=NULL) {
  point_cex = 2.2*cex_val # size of points
  label_cex = 1.25*cex_val # size of axis names
  legend_cex = label_cex
  grid_cex = cex_val
  yaxis_cex = cex_val
  axislabel_line = 4
  if (is.null(problem_types)) {
    problem_types = rep("binary", nrow(results_df))
  }
  if (is.null(official_names)) {
    official_names = names_tocompare
  }
  keep_inds = names_tocompare != autogluon
  names_tocompare = names_tocompare[keep_inds] # remove if it is part of this list
  official_names = official_names[keep_inds]
  alg_names = paste(names_tocompare, suffix, sep="")
  names_tocompare[names_tocompare == 'H2OAutoML'] = 'H2O'
  pch_vals = 18
  method_colors = rep("",length(names_tocompare))
  for (i in 1:length(names_tocompare)) {
    method_colors[i] = adjustcolor(possible_colors[i], alpha = alpha)
  }
  
  # Create plots:
  if (!is.null(savefile)) {
    filestr = tolower(savefile)
    if (substr(filestr, nchar(filestr)-3+1, nchar(filestr)) == 'png') {
      png(savefile, width = 2048, height = 1024)
    } else if (substr(filestr, nchar(filestr)-3+1, nchar(filestr)) == 'pdf') {
      pdf(savefile, width = 24, height = 12)
    } else {
      stop(paste("File-extension in savefile must be either png or pdf, not: ", substr(filestr, nchar(filestr)-3+1, nchar(filestr))))
    }
  }
  # Title is currently ignored:
  # layout(matrix(c(1,1,2,3,4,4), ncol=2, byrow=TRUE), heights=c(1,4, 1))
  # par(mar=c(0,0,0,0))
  # plot.new()
  # text(0.5,0.5,title, cex=2*cex_val, font=2)
  layout(matrix(c(1,2,3,3), ncol=2, byrow=TRUE), heights=c(4, 1))
  
  # Plot performance results:
  y_vals = nrow(results_df):1
  worst_perf = min(apply(results_df, 2, function(x) min(x, na.rm = TRUE)))
  best_perf = max(apply(results_df, 2, function(x) max(x, na.rm = TRUE)))
  x_lim = c(0.99*worst_perf, 1.01*best_perf)
  log_str = ""
  if (log_axis) {
    log_str = "x"
  }
  par(las=1, mar=c(7.0, 18.0, 0.5, 0.5))
  plot(results_df[[1]], y_vals, xlim=x_lim, type='n', yaxt='n', log=log_str, 
       main="", ylab='', xlab='', cex=cex_val, cex.axis=cex_val, cex.lab = label_cex)
  title(xlab = xlabel, cex.lab = label_cex, line = axislabel_line)
  dataset_names = getDatasetNames(results_df)
  axis(side=2,at=y_vals[problem_types == 'binary'],labels = dataset_names[problem_types == 'binary'], cex.axis = yaxis_cex)
  if ('multiclass' %in% problem_types) {
    axis(side=2,at=y_vals[problem_types == 'multiclass'], labels=dataset_names[problem_types == 'multiclass'], 
         col.axis='darkorchid4', cex.axis = yaxis_cex)
  }
  if ('regression' %in% problem_types) {
    axis(side=2,at=y_vals[problem_types == 'regression'], labels=dataset_names[problem_types == 'regression'],
         col.axis = 'darkorange', cex.axis = yaxis_cex)
  }
  ngridpt = ceiling(length(y_vals))
  grid_col = 'snow4'
  grid(ny=0, nx=ngridpt, lwd = grid_cex, col=grid_col)
  for (i in 1:length(y_vals)) {
    abline(h=y_vals[i],lwd = grid_cex,col=grid_col, lty="dotted")
  }
  for (i in 1:length(names_tocompare)) {
    x_vals = results_df[[alg_names[i]]]
    points(x_vals[!is.na(x_vals)], y_vals[!is.na(x_vals)], col = method_colors[i], pch = pch_vals, cex = point_cex)
  }
  abline(v=autogluon_performance, lwd=point_cex*1.25) # Autogluon performance on top
  
  # Plot runtimes:
  min_time = min(apply(time_df, 2, function(x) min(x, na.rm = TRUE)))
  max_time = max(apply(time_df, 2, function(x) max(x, na.rm = TRUE)))
  x_lim = c(0.99*min_time, 1.01*max_time)
  plot(time_df[[1]], y_vals, xlim=x_lim, type='n', yaxt='n', log='x', main="", ylab='', xlab='', cex = cex_val, cex.axis = cex_val, cex.lab = label_cex)
  title(xlab = 'Runtime (min)', cex.lab = label_cex, line = axislabel_line)
  axis(side=2,at=y_vals[problem_types == 'binary'],labels = dataset_names[problem_types == 'binary'], cex.axis = yaxis_cex)
  if ('multiclass' %in% problem_types) {
    axis(side=2,at=y_vals[problem_types == 'multiclass'], labels=dataset_names[problem_types == 'multiclass'], 
         col.axis='darkorchid4', cex.axis = yaxis_cex)
  }
  if ('regression' %in% problem_types) {
    axis(side=2,at=y_vals[problem_types == 'regression'], labels=dataset_names[problem_types == 'regression'],
         col.axis = 'darkorange', cex.axis = yaxis_cex)
  }
  grid(ny=ngridpt, nx=ngridpt, lwd = grid_cex, col='snow4')
  for (i in 1:length(names_tocompare)) {
    points(time_df[[alg_names[i]]], y_vals, col = method_colors[i], pch = pch_vals, cex = point_cex)
  }
  autogluon_name = autogluon_name = paste("autogluon",suffix,sep="")
  points(time_df[[autogluon_name]], y_vals, pch = pch_vals, cex = point_cex)
  
  # Plot legend:
  if (show_legend) {
    par(mai=c(0,0,0,0))
    plot.new()
    official_names = c(official_names, "AutoGluon")
    method_colors = c(method_colors, adjustcolor("black", alpha=alpha))
    legend(x="center", ncol=length(names_tocompare), legend = official_names, col = method_colors, pch=pch_vals, text.col = method_colors, cex = legend_cex)
  }
  if (!is.null(savefile)) {
    print(c("Plot saved in:", savefile))
    dev.off()
  }
}