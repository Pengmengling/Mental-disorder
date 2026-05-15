library(data.table)
library(dplyr)
library(tidyr)
library(readr)

# ============================================================
# 1. Load calibrated risk scores
# ============================================================
input_dir <- "result_calibrated/"
diseases  <- c("F05", "F17", "F32", "F41")

merged <- NULL

for (d in diseases) {
  
  path <- file.path(input_dir, paste0(d, "_calibrated.csv"))
  
  df_tmp <- read_csv(path, show_col_types = FALSE) %>%
    select(`Participant ID`, calibrated_score) %>%
    rename(!!d := calibrated_score)
  
  if (is.null(merged)) {
    merged <- df_tmp
  } else {
    merged <- full_join(merged, df_tmp, by = "Participant ID")
  }
}

risk_data <- merged
setDT(risk_data)

# ============================================================
# 2. Load behavior features and labels
# ============================================================
feature_df <- fread(
  "all_movement_features.csv"
)

feature_df <- feature_df[
  `Participant ID` %in% risk_data$`Participant ID`
]

label_df <- fread(
  "all_icd_f_group_1000_10y_wear.csv"
)

label_df <- label_df[
  `Participant ID` %in% risk_data$`Participant ID`
]

# ============================================================
# 3. Convert behavior features to long format
# ============================================================
feature_long <- feature_df %>%
  pivot_longer(
    cols = -`Participant ID`,
    names_to = "state",
    values_to = "value"
  ) %>%
  mutate(
    state_num  = as.integer(sub("X", "", state)),
    day_type   = ifelse(state_num < 96, "workday", "weekend"),
    pos        = state_num %% 96,
    hour       = pos %/% 4,
    state_type = pos %% 4,
    state_name = case_when(
      state_type == 0 ~ "SB",
      state_type == 1 ~ "LPA",
      state_type == 2 ~ "MVPA",
      state_type == 3 ~ "Sleep"
    )
  ) %>%
  select(-state, -state_num, -pos, -state_type)

setDT(feature_long)

# ============================================================
# 4. Risk stratification
# ============================================================
risk_long <- melt(
  risk_data,
  id.vars = "Participant ID",
  variable.name = "disease",
  value.name = "risk"
)

risk_long[, disease := as.character(disease)]

risk_long[, `:=`(
  low_cut  = quantile(risk, 0.05, na.rm = TRUE),
  high_cut = quantile(risk, 0.95, na.rm = TRUE)
), by = disease]

risk_long[, risk_group := fifelse(
  risk <= low_cut,  "Low",
  fifelse(risk >= high_cut, "High", "Mid")
)]

risk_long <- risk_long[
  !is.na(risk) & !is.na(risk_group)
]

# ============================================================
# 5. Define output settings
# ============================================================
out_dir <- "by_disease_192"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

states <- c("SB", "LPA", "MVPA", "Sleep")
hours  <- 0:23

col_template <- c(
  unlist(lapply(hours, function(h)
    paste0("workday_", h, "_", states))),
  
  unlist(lapply(hours, function(h)
    paste0("weekend_", h, "_", states)))
)

# ============================================================
# 6. Calculate target behavior patterns
# ============================================================
low_pairs <- risk_long[
  risk_group == "Low",
  .(`Participant ID`, disease)
]

feature_low <- feature_long[
  low_pairs,
  on = "Participant ID",
  allow.cartesian = TRUE,
  nomatch = 0
]

low_mean <- feature_low[
  ,
  .(low_mean = mean(value, na.rm = TRUE)),
  by = .(disease, day_type, hour, state_name)
]

high_pairs <- risk_long[
  risk_group == "High",
  .(`Participant ID`, disease)
]

feature_high <- feature_long[
  high_pairs,
  on = "Participant ID",
  allow.cartesian = TRUE,
  nomatch = 0
]

high_mean <- feature_high[
  ,
  .(high_mean = mean(value, na.rm = TRUE)),
  by = .(disease, day_type, hour, state_name)
]

target_pattern <- merge(
  low_mean,
  high_mean,
  by = c("disease", "day_type", "hour", "state_name")
)

target_pattern[, direction := sign(low_mean - high_mean)]

rm(feature_low, feature_high)
gc()

# ============================================================
# 7. Disease-specific adjustment
# ============================================================
adjusted_list <- list()

for (d in diseases) {
  
  cat(sprintf("\nProcessing %s ...\n", d))
  
  pairs_d <- risk_long[
    disease == d,
    .(`Participant ID`, risk_group)
  ]
  
  feature_d <- feature_long[
    pairs_d,
    on = "Participant ID",
    nomatch = 0
  ]
  
  feature_d[, disease := d]
  
  target_d <- target_pattern[
    disease == d,
    .(
      disease,
      day_type,
      hour,
      state_name,
      low_mean,
      direction
    )
  ]
  
  feature_d <- merge(
    feature_d,
    target_d,
    by = c("disease", "day_type", "hour", "state_name"),
    all.x = TRUE
  )
  
  feature_d[, adjusted := fcase(
    
    risk_group == "High",
    0.7 * value + 0.3 * low_mean,
    
    risk_group == "Mid",
    0.7 * value + 0.3 * low_mean,
    
    risk_group == "Low",
    value * (1 + 0.30 * direction)
  )]
  
  wide_d <- dcast(
    feature_d,
    `Participant ID` +
      disease +
      risk_group +
      day_type +
      hour ~ state_name,
    value.var = "adjusted",
    fill = 0
  )
  
  wide_d[, is_night := hour %in% c(22, 23, 0:5)]
  wide_d[, is_day   := hour %in% 8:19]
  
  wide_d[is_night == TRUE, Sleep := pmax(Sleep, 0.70)]
  wide_d[is_night == TRUE, MVPA  := pmin(MVPA, 0.10)]
  wide_d[is_day   == TRUE, SB    := pmin(SB, 0.80)]
  
  wide_d[, total := SB + LPA + MVPA + Sleep]
  
  wide_d[total > 0, `:=`(
    SB    = SB / total,
    LPA   = LPA / total,
    MVPA  = MVPA / total,
    Sleep = Sleep / total
  )]
  
  wide_d[, c("is_night", "is_day", "total") := NULL]
  
  output_d <- dcast(
    wide_d,
    `Participant ID` ~ day_type + hour,
    value.var = c("SB", "LPA", "MVPA", "Sleep")
  )
  
  old_cols <- setdiff(names(output_d), "Participant ID")
  
  setnames(
    output_d,
    old = old_cols,
    new = gsub(
      "(SB|LPA|MVPA|Sleep)_(workday|weekend)_([0-9]+)",
      "\\2_\\3_\\1",
      old_cols
    )
  )
  
  output_d <- output_d[
    ,
    c(col_template, "Participant ID"),
    with = FALSE
  ]
  
  setnames(
    output_d,
    old = names(output_d)[-ncol(output_d)],
    new = as.character(0:191)
  )
  
  fwrite(
    output_d,
    file.path(out_dir, paste0("adjusted_192_", d, ".csv"))
  )
  
  adjusted_list[[d]] <- wide_d[
    ,
    .(
      `Participant ID`,
      disease,
      risk_group,
      day_type,
      hour,
      SB,
      LPA,
      MVPA,
      Sleep
    )
  ]
  
  rm(feature_d, target_d, wide_d, output_d, pairs_d)
  gc()
}

adjusted_all <- rbindlist(adjusted_list)

rm(adjusted_list)
gc()

# ============================================================
# 8. Weighted aggregation across diseases
# ============================================================
case_counts <- label_df[
  ,
  lapply(.SD, function(x) sum(x == 1, na.rm = TRUE)),
  .SDcols = diseases
]

case_counts_long <- melt(
  case_counts,
  measure.vars  = diseases,
  variable.name = "disease",
  value.name    = "n_case"
)

case_counts_long[, weight := n_case / sum(n_case)]
case_counts_long[, disease := as.character(disease)]

adjusted_all[
  case_counts_long,
  on = "disease",
  weight := i.weight
]

adjusted_person <- adjusted_all[
  ,
  .(
    SB    = sum(SB    * weight, na.rm = TRUE),
    LPA   = sum(LPA   * weight, na.rm = TRUE),
    MVPA  = sum(MVPA  * weight, na.rm = TRUE),
    Sleep = sum(Sleep * weight, na.rm = TRUE)
  ),
  by = .(`Participant ID`, day_type, hour)
]

adjusted_192 <- dcast(
  adjusted_person,
  `Participant ID` ~ day_type + hour,
  value.var = c("SB", "LPA", "MVPA", "Sleep")
)

old_cols <- setdiff(names(adjusted_192), "Participant ID")

setnames(
  adjusted_192,
  old = old_cols,
  new = gsub(
    "(SB|LPA|MVPA|Sleep)_(workday|weekend)_([0-9]+)",
    "\\2_\\3_\\1",
    old_cols
  )
)

adjusted_192 <- adjusted_192[
  ,
  c(col_template, "Participant ID"),
  with = FALSE
]

setnames(
  adjusted_192,
  old = names(adjusted_192)[-ncol(adjusted_192)],
  new = as.character(0:191)
)

fwrite(
  adjusted_192,
  "D:/keyan/ICD/xintu/xin_best/all_person_adjusted.csv"
)

cat("\nCompleted successfully.\n")
cat(sprintf("Total participants: %d\n", nrow(adjusted_192)))