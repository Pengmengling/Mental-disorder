library(dplyr)
library(TwoSampleMR)
library(purrr)
library(data.table)

setwd("D:/keyan/ICD/")

# ============================================================
# 1. Paths
# ============================================================
GWAS_DIR <- "PA_GWAS"
LD_DIR <- "LD"
OUTCOME_DIR <- "outcome_gwas"
OUTPUT_DIR <- "MR_results"

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 2. File lists
# ============================================================
gwas_files <- list.files(
  GWAS_DIR,
  pattern = "^assoc\\.regenie\\.merged_.+\\.txt$",
  full.names = TRUE
)

outcome_files <- list.files(
  OUTCOME_DIR,
  pattern = "^finngen_R12_.+\\.gz$",
  full.names = TRUE
)

# ============================================================
# 3. MR pipeline
# ============================================================
results <- map(gwas_files, function(gwas_file) {
  
  file_tag <- gsub("^assoc\\.regenie\\.merged_|\\.txt$", "", basename(gwas_file))
  ld_file <- file.path(LD_DIR, paste0("plink_ld_clumped_", file_tag, ".clumped"))
  
  if (!file.exists(ld_file)) return(NULL)
  
  map(outcome_files, function(outcome_path) {
    
    outcome_tag <- gsub("^finngen_R12_|\\.gz$", "", basename(outcome_path))
    result_dir <- file.path(OUTPUT_DIR, paste0(file_tag, "_vs_", outcome_tag))
    dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)
    
    # ---------------- exposure ----------------
    exposure_gwas <- tryCatch({
      
      lines <- readLines(ld_file, warn = FALSE)
      lines <- trimws(lines)
      lines <- lines[lines != ""]
      lines <- lines[!grepl("^==>", lines)]
      
      ld_snps <- unique(na.omit(sapply(lines, function(x) {
        m <- regexpr("rs[0-9]+", x)
        if (m > 0) substr(x, m, m + attr(m, "match.length") - 1) else NA
      })))
      
      df <- read.table(gwas_file, header = TRUE, stringsAsFactors = FALSE)
      
      snp_col <- intersect(c("SNP", "ID", "MarkerName", "rsid"), names(df))[1]
      if (is.na(snp_col)) return(NULL)
      
      df <- df %>% filter(.data[[snp_col]] %in% ld_snps)
      
      df %>%
        mutate(
          pval = if ("LOG10P" %in% names(.)) 10^(-LOG10P) else as.numeric(pval),
          effect_allele = as.character(ALLELE1),
          other_allele = as.character(ALLELE0)
        ) %>%
        select(
          SNP = all_of(snp_col),
          effect_allele,
          other_allele,
          beta = BETA,
          se = SE,
          pval,
          eaf = A1FREQ
        )
      
    }, error = function(e) NULL)
    
    if (is.null(exposure_gwas) || nrow(exposure_gwas) == 0) return(NULL)
    
    # ---------------- outcome ----------------
    outcome_gwas <- tryCatch({
      
      raw <- read.table(
        gzfile(outcome_path),
        header = TRUE,
        sep = "\t",
        fill = TRUE,
        stringsAsFactors = FALSE
      )
      
      required <- c("rsids", "beta", "sebeta", "pval", "af_alt", "alt", "ref")
      if (!all(required %in% names(raw))) return(NULL)
      
      raw <- raw %>%
        filter(rsids %in% exposure_gwas$SNP)
      
      if (nrow(raw) == 0) return(NULL)
      
      raw %>%
        mutate(
          beta = as.numeric(beta),
          se = as.numeric(sebeta),
          pval = as.numeric(pval),
          eaf = as.numeric(af_alt),
          effect_allele = as.character(alt),
          other_allele = as.character(ref)
        ) %>%
        select(
          SNP = rsids,
          effect_allele,
          other_allele,
          beta,
          se,
          pval,
          eaf
        )
      
    }, error = function(e) NULL)
    
    if (is.null(outcome_gwas) || nrow(outcome_gwas) == 0) return(NULL)
    
    write.csv(exposure_gwas, file.path(result_dir, "exposure.csv"), row.names = FALSE)
    write.csv(outcome_gwas, file.path(result_dir, "outcome.csv"), row.names = FALSE)
    
    # ---------------- MR ----------------
    tryCatch({
      
      exp_dat <- read_exposure_data(file.path(result_dir, "exposure.csv"), sep = ",")
      out_dat <- read_outcome_data(file.path(result_dir, "outcome.csv"), sep = ",")
      
      dat <- harmonise_data(exp_dat, out_dat)
      
      if (nrow(dat) < 3) return(NULL)
      
      mr_res <- mr(dat)
      het_res <- mr_heterogeneity(dat)
      pleio_res <- mr_pleiotropy_test(dat)
      
      write.csv(mr_res, file.path(result_dir, "mr.csv"), row.names = FALSE)
      write.csv(het_res, file.path(result_dir, "heterogeneity.csv"), row.names = FALSE)
      write.csv(pleio_res, file.path(result_dir, "pleiotropy.csv"), row.names = FALSE)
      
      list(
        exposure = file_tag,
        outcome = outcome_tag,
        mr = mr_res,
        heterogeneity = het_res,
        pleiotropy = pleio_res
      )
      
    }, error = function(e) NULL)
    
  }) %>% compact()
  
}) %>% flatten() %>% compact()

# ============================================================
# 4. Summary
# ============================================================
if (length(results) > 0) {
  
  summary_df <- map_dfr(results, function(x) {
    
    ivw <- x$mr[x$mr$method == "Inverse variance weighted", ]
    egger <- x$mr[x$mr$method == "MR Egger", ]
    
    data.frame(
      exposure = x$exposure,
      outcome = x$outcome,
      ivw_beta = ivw$b,
      ivw_p = ivw$pval,
      egger_beta = egger$b,
      egger_p = egger$pval
    )
  })
  
  write.csv(summary_df, file.path(OUTPUT_DIR, "summary.csv"), row.names = FALSE)
}

# ============================================================
# 5. Filtering
# ============================================================
df <- read.csv("MR_all_merged.csv")

df_filtered <- df %>%
  group_by(disease) %>%
  filter(
    length(unique(sign(b[b != 0]))) == 1 &
      any(method == "Inverse variance weighted" & pval < 0.05) &
      any(method == "MR Egger" & pval > 0.05)
  ) %>%
  ungroup()