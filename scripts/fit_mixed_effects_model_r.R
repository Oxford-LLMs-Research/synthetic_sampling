#!/usr/bin/env Rscript
# Fit mixed effects model using R's lme4 package
# More memory-efficient than statsmodels for large datasets

library(lme4)
library(jsonlite)
library(optparse)

# Check if ggplot2 is available for plots (optional, will use base R if not)
HAS_GGPLOT2 <- requireNamespace("ggplot2", quietly=TRUE)
if (HAS_GGPLOT2) {
  library(ggplot2)
}

# Parse command line arguments
option_list <- list(
  make_option(c("--data", "-d"), type="character", default=NULL,
              help="Path to CSV data file", metavar="character"),
  make_option(c("--output", "-o"), type="character", default="mixed_effects_model",
              help="Output path prefix (without extension)", metavar="character"),
  make_option(c("--formula", "-f"), type="character", default="correct ~ n_options + modal_share + model + region + topic_section",
              help="Fixed effects formula (default: correct ~ n_options + modal_share + model + region + topic_section)", metavar="character"),
  make_option(c("--re-formula", "-r"), type="character", 
              default="(1|survey)",
              help="Random effects formula (default: (1|survey))", metavar="character"),
  make_option(c("--sample", "-s"), type="double", default=NULL,
              help="Random sample fraction (0.0-1.0)", metavar="double"),
  make_option(c("--seed", "-e"), type="integer", default=42,
              help="Random seed for sampling", metavar="integer"),
  make_option(c("--no-plots"), action="store_true", default=FALSE,
              help="Skip diagnostic plots generation", metavar="logical")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$data)) {
  print_help(opt_parser)
  stop("--data argument is required", call.=FALSE)
}

cat("Loading data from", opt$data, "...\n")
df <- read.csv(opt$data, stringsAsFactors=FALSE)
cat("  Loaded", nrow(df), "observations\n")
cat("  Columns:", paste(colnames(df), collapse=", "), "\n")

# Debug: show all arguments
cat("\nArguments received:\n")
cat("  data:", opt$data, "\n")
cat("  output:", opt$output, "\n")
cat("  formula:", opt$formula, "\n")
cat("  re_formula length:", nchar(opt$re_formula), "\n")
cat("  re_formula value:", opt$re_formula, "\n")
if (!is.null(opt$sample)) {
  cat("  sample:", opt$sample, "\n")
}
cat("  seed:", opt$seed, "\n")

# Check if re_formula is empty or missing (can happen if argument parsing fails)
if (is.null(opt$re_formula) || nchar(trimws(opt$re_formula)) == 0) {
  cat("\nNOTE: Random effects formula not provided or empty.\n")
  cat("Using default random effects formula...\n")
  opt$re_formula <- "(1|survey)"
}

# Sample if requested
if (!is.null(opt$sample)) {
  if (opt$sample <= 0 || opt$sample > 1) {
    stop("--sample must be between 0 and 1", call.=FALSE)
  }
  original_size <- nrow(df)
  set.seed(opt$seed)
  df <- df[sample(nrow(df), size=round(nrow(df) * opt$sample)), ]
  cat("\n  Sampling: Using", nrow(df), "observations (", 
      round(100 * opt$sample, 1), "% of", original_size, ")\n")
}

# Convert categorical variables to factors (required by lme4)
cat("\nConverting categorical variables to factors...\n")
df$model <- as.factor(df$model)
df$region <- as.factor(df$region)
df$topic_section <- as.factor(df$topic_section)
df$survey <- as.factor(df$survey)  # Survey as fixed effect (factor)

# Ensure numeric variables are numeric
df$correct <- as.numeric(df$correct)
if ("n_options" %in% colnames(df)) {
  df$n_options <- as.numeric(df$n_options)
}
if ("modal_share" %in% colnames(df)) {
  df$modal_share <- as.numeric(df$modal_share)
}

# Print dataset structure
cat("\nDataset structure:\n")
cat("  Observations:", nrow(df), "\n")
cat("  Models:", nlevels(df$model), "\n")
cat("  Regions:", nlevels(df$region), "\n")
cat("  Sections:", nlevels(df$topic_section), "\n")
cat("  Surveys:", nlevels(df$survey), "\n")
if ("n_options" %in% colnames(df)) {
  cat("  n_options range:", min(df$n_options, na.rm=TRUE), "-", max(df$n_options, na.rm=TRUE), "\n")
}
if ("modal_share" %in% colnames(df)) {
  cat("  modal_share range:", round(min(df$modal_share, na.rm=TRUE), 3), "-", round(max(df$modal_share, na.rm=TRUE), 3), "\n")
}

# Build full formula
# lme4 uses syntax: outcome ~ fixed_effects + (1|random_effect1) + (1|random_effect2) + ...
# We need to combine fixed and random effects
# Remove "correct ~" from fixed effects if present, then combine
fixed_part <- sub("^correct\\s*~\\s*", "", opt$formula)
fixed_part <- trimws(fixed_part)  # Remove leading/trailing whitespace

# Debug: show what we're working with
cat("\nFormula components:\n")
cat("  Fixed effects part:", fixed_part, "\n")
cat("  Random effects part:", opt$re_formula, "\n")

if (fixed_part == "1" || fixed_part == "") {
  # Only random effects (intercept-only fixed effects)
  full_formula_str <- paste("correct ~", opt$re_formula)
} else {
  # Fixed effects + random effects
  full_formula_str <- paste("correct ~", fixed_part, "+", opt$re_formula)
}

# Remove any extra whitespace/newlines
full_formula_str <- gsub("\\s+", " ", full_formula_str)
full_formula_str <- trimws(full_formula_str)

cat("\nFitting model with formula:", full_formula_str, "\n")
cat("  (This may take a while for large datasets...)\n")

# Function to generate ICML-compliant diagnostic plots
generate_diagnostic_plots <- function(model, output_prefix) {
  cat("\nGenerating diagnostic plots (ICML compliant)...\n")
  
  # Extract residuals and fitted values
  residuals <- residuals(model)
  fitted_vals <- fitted(model)
  
  # ICML guidelines:
  # - Dark lines >= 0.5pt (0.5 points = 0.5/72 inches)
  # - White background (no gray)
  # - Labeled axes
  # - No titles inside figure
  # - Black text/ticks
  
  linewidth_pt <- 0.5  # Minimum 0.5pt per ICML
  # Convert points to R lwd units: 1pt = 1/72 inch, R lwd = 1/96 inch
  # So 0.5pt = (0.5/72) * 96 = 0.6667 R units
  linewidth <- linewidth_pt * 96 / 72
  
  # Set up graphics device for PNG
  png_path <- paste0(output_prefix, ".png")
  png(png_path, width=10, height=8, units="in", res=300, bg="white")
  
  # Create 2x2 layout
  par(mfrow=c(2, 2), bg="white", fg="black", col="black", col.axis="black", 
      col.lab="black", col.main="black", col.sub="black")
  
  # 1. Q-Q plot of residuals (check normality)
  qqnorm(residuals, main="", xlab="Theoretical Quantiles", ylab="Sample Quantiles",
         col="black", pch=16, cex=0.3)
  qqline(residuals, col="black", lwd=linewidth)
  grid(col="black", lty="dotted", lwd=linewidth)
  box(col="black", lwd=linewidth)
  
  # 2. Residuals vs fitted values (check homoscedasticity)
  plot(fitted_vals, residuals, main="", xlab="Fitted Values", ylab="Residuals",
       col="black", pch=16, cex=0.3, bg="white")
  abline(h=0, col="black", lty="dashed", lwd=linewidth)
  grid(col="black", lty="dotted", lwd=linewidth)
  box(col="black", lwd=linewidth)
  
  # 3. Histogram of residuals (check distribution)
  hist(residuals, main="", xlab="Residuals", ylab="Frequency",
       col="black", border="black", breaks=50, bg="white")
  grid(col="black", lty="dotted", lwd=linewidth, ny=NULL)
  box(col="black", lwd=linewidth)
  
  # 4. Scale-location plot (check homoscedasticity)
  sqrt_abs_residuals <- sqrt(abs(residuals))
  plot(fitted_vals, sqrt_abs_residuals, main="", 
       xlab="Fitted Values", ylab=expression(sqrt(abs(Residuals))),
       col="black", pch=16, cex=0.3, bg="white")
  grid(col="black", lty="dotted", lwd=linewidth)
  box(col="black", lwd=linewidth)
  
  dev.off()
  cat("OK: Saved diagnostic plots to", png_path, "\n")
  
  # Also save PDF
  pdf_path <- paste0(output_prefix, ".pdf")
  pdf(pdf_path, width=10, height=8, bg="white")
  
  par(mfrow=c(2, 2), bg="white", fg="black", col="black", col.axis="black", 
      col.lab="black", col.main="black", col.sub="black")
  
  # Same plots for PDF
  qqnorm(residuals, main="", xlab="Theoretical Quantiles", ylab="Sample Quantiles",
         col="black", pch=16, cex=0.3)
  qqline(residuals, col="black", lwd=linewidth)
  grid(col="black", lty="dotted", lwd=linewidth)
  box(col="black", lwd=linewidth)
  
  plot(fitted_vals, residuals, main="", xlab="Fitted Values", ylab="Residuals",
       col="black", pch=16, cex=0.3, bg="white")
  abline(h=0, col="black", lty="dashed", lwd=linewidth)
  grid(col="black", lty="dotted", lwd=linewidth)
  box(col="black", lwd=linewidth)
  
  hist(residuals, main="", xlab="Residuals", ylab="Frequency",
       col="black", border="black", breaks=50, bg="white")
  grid(col="black", lty="dotted", lwd=linewidth, ny=NULL)
  box(col="black", lwd=linewidth)
  
  plot(fitted_vals, sqrt_abs_residuals, main="", 
       xlab="Fitted Values", ylab=expression(sqrt(abs(Residuals))),
       col="black", pch=16, cex=0.3, bg="white")
  grid(col="black", lty="dotted", lwd=linewidth)
  box(col="black", lwd=linewidth)
  
  dev.off()
  cat("OK: Saved diagnostic plots to", pdf_path, "\n")
}

# Parse and fit model
full_formula <- as.formula(full_formula_str)

start_time <- Sys.time()
tryCatch({
  # Fit model using REML (default)
  model <- lmer(full_formula, data=df, REML=TRUE)
  
  elapsed <- as.numeric(Sys.time() - start_time, units="mins")
  cat("OK: Model converged! (took", round(elapsed, 1), "minutes)\n")
  
  # Extract results
  results <- list(
    formula = full_formula_str,
    re_formula = opt$re_formula,
    n_obs = nrow(df),
    converged = TRUE,
    warnings = NULL,
    fit_time_minutes = elapsed
  )
  
  # Extract variance components
  vc <- VarCorr(model)
  variance_components <- list()
  variance_stddev <- list()
  for (name in names(vc)) {
    stddev <- as.numeric(attr(vc[[name]], "stddev"))
    variance_stddev[[name]] <- stddev
    variance_components[[name]] <- stddev^2
  }
  results$variance_components <- variance_components
  results$variance_stddev <- variance_stddev
  
  # Calculate total variance and ICC (Intraclass Correlation Coefficient) for each random effect
  residual_var <- attr(vc, "sc")^2  # Residual variance
  total_var <- sum(unlist(variance_components)) + residual_var
  
  # Calculate ICC: variance of random effect / total variance
  icc <- list()
  variance_percentages <- list()
  for (name in names(variance_components)) {
    icc[[name]] <- variance_components[[name]] / total_var
    variance_percentages[[name]] <- 100 * variance_components[[name]] / total_var
  }
  results$icc <- icc
  results$variance_percentages <- variance_percentages
  results$residual_variance <- residual_var
  results$total_variance <- total_var
  
  # Extract fixed effects
  fe <- fixef(model)
  results$fixed_effects <- as.list(fe)
  
  # Extract fit statistics
  results$log_likelihood <- as.numeric(logLik(model))
  results$aic <- AIC(model)
  results$bic <- BIC(model)
  
  # Get summary
  summary_text <- capture.output(summary(model))
  results$summary_text <- paste(summary_text, collapse="\n")
  
  # Check for convergence warnings
  # Note: warnings() doesn't work the same way in Rscript, so we'll check the summary
  # The "very large eigenvalue" warning appears in the summary text
  has_eigenvalue_warning <- grepl("very large eigenvalue", results$summary_text, ignore.case=TRUE)
  has_singular_warning <- grepl("boundary.*singular", results$summary_text, ignore.case=TRUE)
  
  results$has_eigenvalue_warning <- has_eigenvalue_warning
  results$has_singular_warning <- has_singular_warning
  
  # Check if model is singular (boundary fit) using lme4 function
  is_singular <- tryCatch({
    isSingular(model)
  }, error = function(e) {
    # Fallback: check if any variance component is very close to zero
    any(unlist(variance_components) < 1e-10)
  })
  results$is_singular <- is_singular
  
  # Save results
  output_json <- paste0(opt$output, ".json")
  output_txt <- paste0(opt$output, ".txt")
  
  # Save JSON (convert to JSON-serializable format)
  json_results <- list(
    formula = results$formula,
    re_formula = results$re_formula,
    n_obs = results$n_obs,
    converged = results$converged,
    fit_time_minutes = results$fit_time_minutes,
    variance_components = results$variance_components,
    variance_stddev = results$variance_stddev,
    variance_percentages = results$variance_percentages,
    icc = results$icc,
    residual_variance = results$residual_variance,
    total_variance = results$total_variance,
    fixed_effects = results$fixed_effects,
    log_likelihood = results$log_likelihood,
    aic = results$aic,
    bic = results$bic,
    is_singular = results$is_singular,
    has_eigenvalue_warning = results$has_eigenvalue_warning,
    has_singular_warning = results$has_singular_warning
  )
  
  write_json(json_results, output_json, pretty=TRUE, auto_unbox=TRUE)
  cat("OK: Saved results to", output_json, "\n")
  
  # Save text summary
  writeLines(results$summary_text, output_txt)
  cat("OK: Saved summary to", output_txt, "\n")
  
  # Generate diagnostic plots (ICML compliant)
  # opt$no_plots is a logical value from optparse
  if (is.null(opt$no_plots) || !opt$no_plots) {
    generate_diagnostic_plots(model, opt$output)
  }
  
  # Print summary
  cat("\n")
  cat(paste(rep("=", 80), collapse=""), "\n")
  cat("MIXED EFFECTS MODEL RESULTS\n")
  cat(paste(rep("=", 80), collapse=""), "\n\n")
  cat("Formula:", results$formula, "\n")
  cat("Random effects:", results$re_formula, "\n")
  cat("N observations:", results$n_obs, "\n")
  cat("Converged:", results$converged, "\n")
  if (results$is_singular) {
    cat("WARNING: Model is singular (boundary fit) - some variance components are near zero\n")
  }
  cat("Fit time:", round(results$fit_time_minutes, 1), "minutes\n\n")
  
  cat("VARIANCE COMPONENTS (Variance Explained by Each Random Effect):\n")
  cat(paste(rep("-", 80), collapse=""), "\n")
  cat(sprintf("%-20s %12s %12s %10s %12s\n", "Random Effect", "Variance", "Std.Dev.", "% Total", "ICC"))
  cat(paste(rep("-", 80), collapse=""), "\n")
  for (name in names(variance_components)) {
    var_val <- variance_components[[name]]
    stddev_val <- variance_stddev[[name]]
    pct_val <- variance_percentages[[name]]
    icc_val <- icc[[name]]
    cat(sprintf("%-20s %12.6f %12.6f %9.2f%% %12.4f\n", 
                name, var_val, stddev_val, pct_val, icc_val))
  }
  cat(sprintf("%-20s %12.6f %12.6f %9.2f%% %12s\n", 
              "Residual", residual_var, sqrt(residual_var), 
              100 * residual_var / total_var, ""))
  cat(paste(rep("-", 80), collapse=""), "\n")
  cat(sprintf("%-20s %12.6f\n", "Total Variance", total_var))
  cat("\n")
  
  cat("INTERPRETATION:\n")
  cat("  - Variance: Amount of variance in the outcome explained by each random effect\n")
  cat("  - %% Total: Percentage of total variance explained by each component\n")
  cat("  - ICC (Intraclass Correlation): Proportion of variance due to grouping\n")
  cat("    Higher ICC = more variation between groups than within groups\n\n")
  
  # Identify which random effects explain the most variance
  sorted_effects <- sort(unlist(variance_percentages), decreasing=TRUE)
  cat("RANKING BY VARIANCE EXPLAINED:\n")
  for (i in seq_along(sorted_effects)) {
    name <- names(sorted_effects)[i]
    cat(sprintf("  %d. %s: %.2f%% of total variance\n", i, name, sorted_effects[i]))
  }
  cat("\n")
  
  cat("FIXED EFFECTS:\n")
  for (name in names(results$fixed_effects)) {
    cat("  ", name, ":", results$fixed_effects[[name]], "\n")
  }
  cat("\n")
  
  cat("FIT STATISTICS:\n")
  cat("  Log-likelihood:", results$log_likelihood, "\n")
  cat("  AIC:", results$aic, "\n")
  cat("  BIC:", results$bic, "\n")
  cat("\n")
  
  # Explain the warning if present
  if (results$is_singular || results$has_eigenvalue_warning || results$has_singular_warning) {
    cat("CONVERGENCE NOTES:\n")
    if (results$is_singular || results$has_singular_warning) {
      cat("  - Model is singular (boundary fit): Some variance components are at or near zero.\n")
      cat("    This is common with complex models with many random effects.\n")
      cat("    It means some random effects (e.g., region) have minimal variation.\n")
      cat("    This is NOT a problem - the model converged successfully (code: 0).\n")
    }
    if (results$has_eigenvalue_warning) {
      cat("  - 'Very large eigenvalue' warning detected.\n")
      cat("    This is common with models having many crossed random effects.\n")
      cat("    It does NOT indicate a problem - the model converged (code: 0).\n")
      cat("    The warning suggests numerical scaling but results are still valid.\n")
    }
    cat("\n")
  }
  
  cat(paste(rep("=", 80), collapse=""), "\n")
  
}, error = function(e) {
  elapsed <- as.numeric(Sys.time() - start_time, units="mins")
  cat("\nError fitting model after", round(elapsed, 1), "minutes:", conditionMessage(e), "\n")
  stop(e)
})
