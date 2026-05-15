library(dplyr)
library(tidyr)
library(purrr)
library(lavaan)
library(ggplot2)

# =========================================================
# 1. Basic settings
# =========================================================

states <- c("Sedentary", "Light", "Moderate-Vigorous", "Sleep")
disease_list <- c("F05", "F17", "F32", "F41")

# =========================================================
# 2. Load data
# =========================================================

pa <- read.csv("pa_exposure_mediator_results.csv")

final_df <- read.csv("pro_disease3.csv")

df_pro <- read.csv("master_pro.csv")

event_time <- read.csv("label.csv")

df_activity <- read.csv("all_movement_features.csv")

# =========================================================
# 3. Utility functions
# =========================================================

clean_name <- function(x) {
  x %>%
    gsub("[^A-Za-z0-9]+", ".", .) %>%
    gsub("\\.+", ".", .) %>%
    gsub("^\\.|\\.$", "", .)
}

build_mediation_model <- function(mediator) {
  
  paste0(
    mediator, " ~ a*X\n",
    "Y ~ b*", mediator, " + cprime*X\n",
    "indirect := a*b\n",
    "total := cprime + (a*b)"
  )
}

# =========================================================
# 4. Protein preprocessing
# =========================================================

df_pro <- df_pro %>%
  select(
    Participant.ID,
    c(2:which(colnames(df_pro) ==
                "MYBPC1.Myosin.binding.protein.C..slow.type"))
  )

colnames(df_pro)[-1] <- clean_name(colnames(df_pro)[-1])

df_pro <- df_pro %>%
  filter(!if_all(-1, is.na))

# =========================================================
# 5. Activity preprocessing
# =========================================================

df_long <- df_activity %>%
  pivot_longer(
    cols = -Participant.ID,
    names_to = "state",
    values_to = "value"
  ) %>%
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
  summarise(
    time = sum(value) / 48 * 24,
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = state_name,
    values_from = time
  )

# =========================================================
# 6. Merge analysis dataset
# =========================================================

df_sem <- df_pro %>%
  left_join(
    event_time %>%
      select(Participant.ID, all_of(disease_list)),
    by = "Participant.ID"
  ) %>%
  left_join(activity_wide, by = "Participant.ID")

# =========================================================
# 7. Significant mediator filtering
# =========================================================

final_df <- final_df %>%
  mutate(FDR = p.adjust(p, method = "BH"))

sig_df <- final_df %>%
  filter(FDR < 0.05) %>%
  mutate(feature_clean = clean_name(feature))

pa_sig <- pa %>%
  filter(FDR < 0.05) %>%
  mutate(
    mediator_clean = clean_name(mediator)
  )

state_map <- c(
  "Sedentary - Overall average | Instance 0" =
    "Sedentary",
  
  "Moderate-Vigorous - Overall average | Instance 0" =
    "Moderate-Vigorous",
  
  "Light - Overall average | Instance 0" =
    "Light",
  
  "Sleep - Overall average | Instance 0" =
    "Sleep"
)

pa_sig <- pa_sig %>%
  mutate(
    exposure_clean = state_map[exposure]
  )

# =========================================================
# 8. Obtain intersected mediators
# =========================================================

get_intersection_by_disease_state <- function(dis, state_name) {
  
  disease_mediators <- sig_df %>%
    filter(disease == dis) %>%
    pull(feature_clean)
  
  activity_mediators <- pa_sig %>%
    filter(exposure_clean == state_name) %>%
    pull(mediator_clean)
  
  intersect(disease_mediators, activity_mediators)
}

# =========================================================
# 9. Mediation analysis
# =========================================================

results_all <- list()

for(dis in disease_list){
  
  message("Processing disease: ", dis)
  
  df_sem$Y <- df_sem[[dis]]
  
  for(st in states){
    
    message("  Activity state: ", st)
    
    df_sem$X <- as.numeric(scale(df_sem[[st]]))
    
    mediators_raw <- get_intersection_by_disease_state(dis, st)
    
    mediators_use <- mediators_raw[
      mediators_raw %in% colnames(df_pro)
    ]
    
    if(length(mediators_use) == 0){
      next
    }
    
    for(med in mediators_use){
      
      model_string <- build_mediation_model(med)
      
      fit <- tryCatch(
        
        sem(
          model_string,
          data = df_sem,
          ordered = "Y",
          estimator = "WLSMV"
        ),
        
        error = function(e) NULL
      )
      
      if(is.null(fit)){
        next
      }
      
      res <- tryCatch(
        
        parameterEstimates(
          fit,
          standardized = TRUE
        ) %>%
          mutate(
            Disease = dis,
            State = st,
            Mediator = med
          ),
        
        error = function(e) NULL
      )
      
      if(is.null(res)){
        next
      }
      
      results_all[[paste(
        dis,
        st,
        med,
        sep = "_"
      )]] <- res
    }
  }
}

# =========================================================
# 10. Merge results
# =========================================================

results_df <- bind_rows(results_all)

# =========================================================
# 11. Significant mediation results
# =========================================================

labels_need <- c(
  "a",
  "b",
  "cprime",
  "indirect",
  "total"
)

results_sig <- results_df %>%
  filter(label %in% labels_need) %>%
  group_by(Disease, State, Mediator) %>%
  filter(all(labels_need %in% label)) %>%
  filter(all(pvalue < 0.05)) %>%
  ungroup()


