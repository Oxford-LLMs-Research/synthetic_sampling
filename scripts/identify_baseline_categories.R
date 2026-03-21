#!/usr/bin/env Rscript
# Identify baseline/reference categories for fixed effects in mixed effects model
# The baseline is the first level alphabetically (not shown in coefficients)

library(optparse)

option_list <- list(
  make_option(c("--data", "-d"), type="character", default=NULL,
              help="Path to CSV data file", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$data)) {
  print_help(opt_parser)
  stop("--data argument is required", call.=FALSE)
}

cat("Loading data from", opt$data, "...\n")
df <- read.csv(opt$data, stringsAsFactors=FALSE)
cat("  Loaded", nrow(df), "observations\n\n")

# Convert to factors to see level ordering
df$model <- as.factor(df$model)
df$region <- as.factor(df$region)
df$topic_section <- as.factor(df$topic_section)

cat(paste(rep("=", 80), collapse=""), "\n")
cat("BASELINE/REFERENCE CATEGORIES FOR FIXED EFFECTS\n")
cat(paste(rep("=", 80), collapse=""), "\n\n")
cat("In R's lme4, the first factor level (alphabetically) is used as the baseline.\n")
cat("All other levels are compared to this baseline.\n\n")

cat("MODELS:\n")
cat("  Total models:", nlevels(df$model), "\n")
cat("  All model levels (alphabetical order):\n")
model_levels <- levels(df$model)
for (i in seq_along(model_levels)) {
  marker <- if (i == 1) "  -> " else "     "
  cat(sprintf("%s%d. %s%s\n", marker, i, model_levels[i], if (i == 1) " [BASELINE - not shown in coefficients]" else ""))
}
cat("\n")

cat("REGIONS:\n")
cat("  Total regions:", nlevels(df$region), "\n")
cat("  All region levels (alphabetical order):\n")
region_levels <- levels(df$region)
for (i in seq_along(region_levels)) {
  marker <- if (i == 1) "  -> " else "     "
  cat(sprintf("%s%d. %s%s\n", marker, i, region_levels[i], if (i == 1) " [BASELINE - not shown in coefficients]" else ""))
}
cat("\n")

cat("TOPIC SECTIONS:\n")
cat("  Total topic sections:", nlevels(df$topic_section), "\n")
cat("  All topic section levels (alphabetical order):\n")
topic_levels <- levels(df$topic_section)
for (i in seq_along(topic_levels)) {
  marker <- if (i == 1) "  -> " else "     "
  cat(sprintf("%s%d. %s%s\n", marker, i, topic_levels[i], if (i == 1) " [BASELINE - not shown in coefficients]" else ""))
}
cat("\n")

cat(paste(rep("=", 80), collapse=""), "\n")
cat("SUMMARY:\n")
cat(paste(rep("=", 80), collapse=""), "\n")
cat(sprintf("Baseline MODEL: %s\n", model_levels[1]))
cat(sprintf("Baseline REGION: %s\n", region_levels[1]))
cat(sprintf("Baseline TOPIC_SECTION: %s\n", topic_levels[1]))
cat("\n")
cat("All coefficients in your model output are relative to these baselines.\n")
cat("For example, 'modelgemma-3-27b-instruct' coefficient shows the difference\n")
cat("from the baseline model (", model_levels[1], ").\n", sep="")
