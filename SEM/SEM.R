library(dplyr)
library(tidyr)
library(lavaan)

states <- c("Sedentary", "Light", "Moderate-Vigorous", "Sleep")
disease_list <- c("F17","F32","F41","F05")

pa <- read.csv("D:/keyan/ICD/PA中介分析/PA中介分析（通用）/pa_exposure_mediator_results(bb+bp+nmr).csv")
final_df <- read.csv("D:/keyan/ICD/xintu/protein_bb+bp+nmr/bb+bp+nmr_disease3.csv")
df_pro <- read.csv("D:/keyan/ICD/PA中介分析/PA中介分析（通用）/master_bb+bp+nmr.csv")
event_time <- read.csv("D:/keyan/UKB/data/label.csv")
df <- read.csv("D:/keyan/ICD/data/feature and lable/all_movement_features.csv")

clean_feature <- function(x) {
  x %>%
    gsub("\\|? *Instance[ .]?0$", "", .) %>%
    gsub("\\|", "", .) %>%
    gsub(" +", ".", .) %>%
    gsub("\\.+", ".", .) %>%
    gsub("^\\.+|\\.+$", "", .) %>%
    trimws()
}

df_pro <- df_pro %>%
  select(Participant.ID, c(which(colnames(df_pro)=="Free.Cholesterol.in.Large.LDL...Instance.0"):which(colnames(df_pro)=="Basophill.count...Instance.0")))
colnames(df_pro)[-1] <- clean_feature(colnames(df_pro)[-1])
df_pro_clean <- df_pro %>% filter(!if_all(-1, is.na))

df_long <- df %>%
  pivot_longer(cols = -Participant.ID, names_to = "state", values_to = "value") %>%
  mutate(
    state_num = as.integer(sub("X","",state)),
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
  summarise(time = sum(value)/48*24, .groups="drop") %>%
  pivot_wider(names_from = state_name, values_from = time)

df_sem <- df_pro_clean %>%
  left_join(event_time %>% select(Participant.ID, all_of(disease_list)), by="Participant.ID") %>%
  left_join(activity_wide, by="Participant.ID")

sig_df <- final_df %>% filter(FDR < 0.05) %>% mutate(feature_clean = clean_feature(feature))
pa_sig <- pa %>% filter(FDR < 0.05) %>% mutate(mediator_clean = clean_feature(mediator))

state_map <- c(
  "Sedentary - Overall average | Instance 0" = "Sedentary",
  "Moderate-Vigorous - Overall average | Instance 0" = "Moderate-Vigorous",
  "Light - Overall average | Instance 0" = "Light",
  "Sleep - Overall average | Instance 0" = "Sleep"
)

pa_sig <- pa_sig %>% mutate(exposure_clean = state_map[exposure])

get_intersection_by_disease_state <- function(dis, state_name){
  genes_dis <- sig_df %>% filter(disease == dis) %>% pull(feature_clean)
  mediators_state <- pa_sig %>% filter(exposure_clean == state_name) %>% pull(mediator_clean)
  intersect(genes_dis, mediators_state)
}

build_single_mediation_model <- function(m){
  paste0(
    m, " ~ a*X\n",
    "Y ~ b*", m, " + cprime*X\n",
    "indirect := a*b\n",
    "total := cprime + (a*b)"
  )
}

disease_list <- c("F17","F32","F41","F05")
df_sem <- df_pro %>%
  left_join(event_time %>% select(Participant.ID, all_of(disease_list)), by="Participant.ID") %>%
  left_join(activity_wide, by="Participant.ID")

results_all <- list()

for(dis in disease_list){
  df_sem$Y <- df_sem[[dis]]
  for(st in states){
    df_sem$X <- as.numeric(scale(df_sem[[st]]))
    mediators_raw <- get_intersection_by_disease_state(dis, st)
    mediators_use <- mediators_raw[mediators_raw %in% colnames(df_pro)]
    if(length(mediators_use) == 0) next
    for(med in mediators_use){
      model_string <- build_single_mediation_model(med)
      fit <- tryCatch(
        sem(model_string, data = df_sem, ordered = "Y", estimator = "WLSMV"),
        error = function(e) NULL
      )
      if(is.null(fit)) next
      res <- tryCatch(
        parameterEstimates(fit, standardized = TRUE) %>%
          dplyr::mutate(Disease = dis, State = st, Mediator = med),
        error = function(e) NULL
      )
      if(is.null(res)) next
      results_all[[paste(dis, st, med, sep="_")]] <- res
    }
  }
}

results_df <- bind_rows(results_all)

