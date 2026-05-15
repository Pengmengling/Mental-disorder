library(dplyr)
library(tidyr)
library(lavaan)

# =========================
# 1. Settings
# =========================

states <- c("Sedentary", "Light", "Moderate-Vigorous", "Sleep")
disease_list <- c("F17", "F32", "F41", "F05")

# =========================
# 2. Load data
# =========================

pa <- read.csv("pa_exposure_mediator_results.csv")
final_df <- read.csv("bbbp_nmr_disease3.csv")
df_metab <- read.csv("master_bb_bp_nmr.csv")
event_time <- read.csv("label.csv")
df <- read.csv("all_movement_features.csv")

# =========================
# 3. Cleaning function
# =========================

clean_feature <- function(x) {
  x %>%
    gsub("\\|? *Instance[ .]?0$", "", .) %>%
    gsub("\\|", "", .) %>%
    gsub(" +", ".", .) %>%
    gsub("\\.+", ".", .) %>%
    gsub("^\\.+|\\.+$", "", .) %>%
    trimws()
}

# =========================
# 4. Inflammatory-metabolic feature preprocessing
# =========================

df_metab <- df_metab %>%
  select(Participant.ID,
         Free.Cholesterol.in.Large.LDL...Instance.0:Basophill.count...Instance.0)

colnames(df_metab)[-1] <- clean_feature(colnames(df_metab)[-1])

df_metab <- df_metab %>%
  filter(!if_all(-1, is.na))

# =========================
# 5. Activity processing
# =========================

df_long <- df %>%
  pivot_longer(-Participant.ID, names_to = "state", values_to = "value") %>%
  mutate(
    state_num = as.integer(sub("X", "", state)),
    pos = state_num %% 96,
    state_type = pos %% 4,
    state_name = case_when(
      state_type == 0 ~ "Sedentary",
      state_type == 1 ~ "Light",
      state_type == 2 ~ "Moderate-Vigorous",
      state_type == 3 ~ "Sleep"
    )
  ) %>%
  select(Participant.ID, state_name, value)

activity_wide <- df_long %>%
  group_by(Participant.ID, state_name) %>%
  summarise(time = sum(value) / 48 * 24, .groups = "drop") %>%
  pivot_wider(names_from = state_name, values_from = time)

# =========================
# 6. Merge dataset
# =========================

df_sem <- df_metab %>%
  left_join(event_time %>% select(Participant.ID, all_of(disease_list)),
            by = "Participant.ID") %>%
  left_join(activity_wide, by = "Participant.ID")

# =========================
# 7. FDR filtering (inflammatory-metabolic mediators)
# =========================

sig_df <- final_df %>%
  filter(FDR < 0.05) %>%
  mutate(inflam_metab_feature = clean_feature(feature))

pa_sig <- pa %>%
  filter(FDR < 0.05) %>%
  mutate(inflam_metab_mediator = clean_feature(mediator))

state_map <- c(
  "Sedentary - Overall average | Instance 0" = "Sedentary",
  "Light - Overall average | Instance 0" = "Light",
  "Moderate-Vigorous - Overall average | Instance 0" = "Moderate-Vigorous",
  "Sleep - Overall average | Instance 0" = "Sleep"
)

pa_sig <- pa_sig %>%
  mutate(exposure_clean = state_map[exposure])

# =========================
# 8. Intersection function
# =========================

get_intersection_by_disease_state <- function(dis, state_name) {
  
  inflam_metab_features <- sig_df %>%
    filter(disease == dis) %>%
    pull(inflam_metab_feature)
  
  inflam_metab_mediators <- pa_sig %>%
    filter(exposure_clean == state_name) %>%
    pull(inflam_metab_mediator)
  
  intersect(inflam_metab_features, inflam_metab_mediators)
}

# =========================
# 9. SEM model
# =========================

build_model <- function(m) {
  paste0(
    m, " ~ a*X\n",
    "Y ~ b*", m, " + cprime*X\n",
    "indirect := a*b\n",
    "total := cprime + (a*b)"
  )
}

# =========================
# 10. Mediation loop
# =========================

results_all <- list()

for (dis in disease_list) {
  
  df_sem$Y <- df_sem[[dis]]
  
  for (st in states) {
    
    df_sem$X <- as.numeric(scale(df_sem[[st]]))
    
    mediators_raw <- get_intersection_by_disease_state(dis, st)
    mediators_use <- mediators_raw[mediators_raw %in% colnames(df_metab)]
    
    if (length(mediators_use) == 0) next
    
    for (med in mediators_use) {
      
      model <- build_model(med)
      
      fit <- tryCatch(
        sem(model,
            data = df_sem,
            ordered = "Y",
            estimator = "WLSMV"),
        error = function(e) NULL
      )
      
      if (is.null(fit)) next
      
      res <- parameterEstimates(fit, standardized = TRUE) %>%
        mutate(
          Disease = dis,
          State = st,
          Inflam_Metab_Mediator = med
        )
      
      results_all[[paste(dis, st, med, sep = "_")]] <- res
    }
  }
}

# =========================
# 11. Output
# =========================

results_df <- bind_rows(results_all)

write.csv(results_df, "inflammatory_metabolic_mediation_results.csv", row.names = FALSE)