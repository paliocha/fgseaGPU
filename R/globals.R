# Silence R CMD check NOTEs about data.table NSE column references in
# setorder() / setattr() calls inside fgsea.R. These names are evaluated
# in the data.table's frame, not in the function's environment.

utils::globalVariables(c("pval", "ES", "hog_idx"))
