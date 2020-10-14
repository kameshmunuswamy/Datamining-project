#Loading the packages
install.packages("gmodels")
library(gmodels) # Cross Tables [CrossTable()]
install.packages("ggmosaic")
library(ggmosaic) # Mosaic plot with ggplot [geom_mosaic()]
install.packages("corrplot")
library(corrplot) # Correlation plot [corrplot()]
install.packages("ggpubr")
library(ggpubr) # Arranging ggplots together [ggarrange()]
install.packages("cowplot")
library(cowplot) # Arranging ggplots together [plot_grid()]
install.packages("caret")
library(caret) # ML [train(), confusionMatrix(), createDataPartition(), varImp(), trainControl()]
install.packages("ROCR")
library(ROCR) # Model performance [performance(), prediction()]
install.packages("plotROC")
library(plotROC) # ROC Curve with ggplot [geom_roc()]
install.packages("pROC")
library(pROC) # AUC computation [auc()]
install.packages("PRROC")
library(PRROC) # AUPR computation [pr.curve()]
install.packages("rpart")
library(rpart) # Decision trees [rpart(), plotcp(), prune()]
install.packages("rpart.plot")
library(rpart.plot) # Decision trees plotting [rpart.plot()]
install.packages("ranger")
library(ranger) # Optimized Random Forest [ranger()]
install.packages("lightgbm")
library(lightgbm) # Light GBM [lgb.train()]
install.packages("xgboost")
library(xgboost) # XGBoost [xgb.DMatrix(), xgb.train()]
install.packages("MLmetrics")
library(MLmetrics) # Custom metrics (F1 score for example)
install.packages("tidyverse")
library(tidyverse) # Data manipulation
install.packages("dplyr")
library(dplyr)
install.packages("tidyr")
library(tidyr)
install.packages("vegtable")
library(vegtable)
install.packages("descr")
library(descr)
install.packages("xtable")
library(xtable)

#importing the data 
bank_data = read.csv(file = "C:/Users/Kamesh/Desktop/Data mining project dataset/bank dataset/bank-additional-full.csv",
                     sep = ",",
                     stringsAsFactors = F)


dim(bank_data)


names(bank_data)


CrossTable(bank_data$y)


bank_data = bank_data %>% 
  mutate(y = factor(if_else(y == "yes", "1", "0"), 
                    levels = c("0", "1")))

head(bank_data)

#Checking for Null values 
sum(is.na(bank_data))

#cHecking for Unknow values
sum(bank_data == "unknown")

bank_data %>% 
  summarise_all(list(~sum(. == "unknown"))) %>% 
  gather(key = "variable", value = "nr_unknown") %>% 
  arrange(-nr_unknown)


##Cross validation logic
# default theme for ggplot
theme_set(theme_bw())

# setting default parameters for mosaic plots
mosaic_theme = theme(axis.text.x = element_text(angle = 90,
                                                hjust = 1,
                                                vjust = 0.5),
                     axis.text.y = element_blank(),
                     axis.ticks.y = element_blank())




# setting default parameters for crosstables
fun_crosstable = function(df, var1, var2){
  # df: dataframe containing both columns to cross
  # var1, var2: columns to cross together.
  CrossTable(df[, var1], df[, var2],
             prop.r = T,
             prop.c = F,
             prop.t = F,
             prop.chisq = F,
             dnn = c(var1, var2))
}

# plot weighted lm/leoss regressions with frequencies
fun_gg_freq = function(var){
  # var: which column from bank_data to use in regressions
  
  # computing weights first...
  weight = table(bank_data[, var]) %>% 
    as.data.frame %>% 
    mutate(x = as.numeric(as.character(Var1))) %>% 
    select(-Var1) %>% 
    rename(weight = Freq)
  
  # ... then frequencies
  sink(tempfile())
  freq = fun_crosstable(bank_data, var, "y")$prop.r %>% 
    as.data.frame %>% 
    mutate(x = as.numeric(as.character(x)))
  sink()
  
  # assembling
  both = freq %>% 
    left_join(weight, by = "x") %>% 
    filter(weight > 50 & y == 1)
  
  # plotting
  gg = both %>% 
    ggplot() +
    aes(x = x,
        y = Freq,
        weight = weight) +
    geom_point(aes(size = weight)) +
    geom_smooth(aes(colour = "blue"), method = "loess") +
    geom_smooth(aes(colour = "red"), method = "lm", se = F) +
    coord_cartesian(ylim = c(-0.1, 1)) +
    theme(plot.margin = unit(c(0, 0, 0, 0), "pt")) +
    xlab(var) +
    ylab("") +
    scale_x_continuous(position = "top") +
    scale_colour_manual(values = c("blue", "red"),
                        labels = c("loess", "lm")) +
    labs(colour = "Regression")
  
  return(gg)
}

# re-ordering levels from factor variable
fun_reorder_levels = function(df, variable, first){
  # df: dataframe containing columns to transform into factors
  # variable: variable to transform into factor
  # first: first level of the variable to transform.
  
  remaining = unique(df[, variable])[which(unique(df[, variable]) != first)]
  x = factor(df[, variable], levels = c(first, remaining))
  return(x)
}

# plotting importance from predictive models into two panels
fun_imp_ggplot_split = function(model){
  # model: model used to plot variable importances
  
  if (class(model)[1] == "ranger"){
    imp_df = model$variable.importance %>% 
      data.frame("Overall" = .) %>% 
      rownames_to_column() %>% 
      rename(variable = rowname) %>% 
      arrange(-Overall)
  } else {
    imp_df = varImp(model) %>%
      rownames_to_column() %>% 
      rename(variable = rowname) %>% 
      arrange(-Overall)
  }
  
  # first panel (half most important variables)
  gg1 = imp_df %>% 
    slice(1:floor(nrow(.)/2)) %>% 
    ggplot() +
    aes(x = reorder(variable, Overall), weight = Overall, fill = -Overall) +
    geom_bar() +
    coord_flip() +
    xlab("Variables") +
    ylab("Importance") +
    theme(legend.position = "none")
  
  imp_range = ggplot_build(gg1)[["layout"]][["panel_params"]][[1]][["x.range"]]
  imp_gradient = scale_fill_gradient(limits = c(-imp_range[2], -imp_range[1]),
                                     low = "#132B43", 
                                     high = "#56B1F7")
  
  # second panel (less important variables)
  gg2 = imp_df %>% 
    slice(floor(nrow(.)/2)+1:nrow(.)) %>% 
    ggplot() +
    aes(x = reorder(variable, Overall), weight = Overall, fill = -Overall) +
    geom_bar() +
    coord_flip() +
    xlab("") +
    ylab("Importance") +
    theme(legend.position = "none") +
    ylim(imp_range) +
    imp_gradient
  
  # arranging together
  gg_both = plot_grid(gg1 + imp_gradient,
                      gg2)
  
  return(gg_both)
}

# plotting two performance measures
fun_gg_cutoff = function(score, obs, measure1, measure2) {
  # score: predicted scores
  # obs: real classes
  # measure1, measure2: which performance metrics to plot
  
  predictions = prediction(score, obs)
  performance1 = performance(predictions, measure1)
  performance2 = performance(predictions, measure2)
  
  df1 = data.frame(x = performance1@x.values[[1]],
                   y = performance1@y.values[[1]],
                   measure = measure1,
                   stringsAsFactors = F) %>% 
    drop_na()
  df2 = data.frame(x = performance2@x.values[[1]],
                   y = performance2@y.values[[1]],
                   measure = measure2,
                   stringsAsFactors = F) %>% 
    drop_na()
  
  # df contains all the data needed to plot both curves
  df = df1 %>% 
    bind_rows(df2)
  
  # extracting best cut for each measure
  y_max_measure1 = max(df1$y, na.rm = T)
  x_max_measure1 = df1[df1$y == y_max_measure1, "x"][1]
  
  y_max_measure2 = max(df2$y, na.rm = T)
  x_max_measure2 = df2[df2$y == y_max_measure2, "x"][1]
  
  txt_measure1 = paste("Best cut for", measure1, ": x =", round(x_max_measure1, 3))
  txt_measure2 = paste("Best cut for", measure2, ": x =", round(x_max_measure2, 3))
  txt_tot = paste(txt_measure1, "\n", txt_measure2, sep = "")
  
  # plotting both measures in the same plot, with some detail around.
  gg = df %>% 
    ggplot() +
    aes(x = x,
        y = y,
        colour = measure) +
    geom_line() +
    geom_vline(xintercept = c(x_max_measure1, x_max_measure2), linetype = "dashed", color = "gray") +
    geom_hline(yintercept = c(y_max_measure1, y_max_measure2), linetype = "dashed", color = "gray") +
    labs(caption = txt_tot) +
    theme(plot.caption = element_text(hjust = 0)) +
    xlim(c(0, 1)) +
    ylab("") +
    xlab("Threshold")
  
  return(gg)
}

# creating classes according to score and cut
fun_cut_predict = function(score, cut) {
  # score: predicted scores
  # cut: threshold for classification
  
  classes = score
  classes[classes > cut] = 1
  classes[classes <= cut] = 0
  classes = as.factor(classes)
  
  return(classes)  
}

# computing AUPR
aucpr = function(obs, score){
  # obs: real classes
  # score: predicted scores
  
  df = data.frame("pred" = score,
                  "obs" = obs)
  
  prc = pr.curve(df[df$obs == 1, ]$pred,
                 df[df$obs == 0, ]$pred)
  
  return(prc$auc.davis.goadrich)
}

# plotting PR curve
gg_prcurve = function(df) {
  # df: df containing models scores by columns and the last column must be
  #     nammed "obs" and must contain real classes.
  
  # init
  df_gg = data.frame("v1" = numeric(), 
                     "v2" = numeric(), 
                     "v3" = numeric(), 
                     "model" = character(),
                     stringsAsFactors = F)
  
  # individual pr curves
  for (i in c(1:(ncol(df)-1))) {
    x1 = df[df$obs == 1, i]
    x2 = df[df$obs == 0, i]
    prc = pr.curve(x1, x2, curve = T)
    
    df_prc = as.data.frame(prc$curve, stringsAsFactors = F) %>% 
      mutate(model = colnames(df)[i])
    
    # combining pr curves
    df_gg = bind_rows(df_gg,
                      df_prc)
    
  }
  
  gg = df_gg %>% 
    ggplot() +
    aes(x = V1, y = V2, colour = model) +
    geom_line() +
    xlab("Recall") +
    ylab("Precision")
  
  return(gg)
}

summary(bank_data$age)


bank_data %>% 
  ggplot() +
  aes(x = age) +
  geom_bar() +
  geom_vline(xintercept = c(30, 60), 
             col = "red",
             linetype = "dashed") +
  facet_grid(y ~ .,
             scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 100, 5))


bank_data %>% 
  mutate(elder60 = if_else(age > 60, "1", "0")) %>% 
  group_by(y) %>% 
  add_count(nr_y = n()) %>% 
  group_by(elder60, y) %>% 
  summarise(abs_freq = n(),
            relative_freq = round(100*n()/first(nr_y), 2))



bank_data = bank_data %>% 
  mutate(age = if_else(age > 60, "high", if_else(age > 30, "mid", "low")))

#age
fun_crosstable(bank_data, "age", "y")

#job
table(bank_data$job)

fun_crosstable(bank_data, "job", "y")


bank_data = bank_data %>% 
  filter(job != "unknown")

#marital situation

fun_crosstable(bank_data, "marital", "y")

bank_data = bank_data %>% 
  filter(marital != "unknown")


bank_data %>% 
  ggplot() +
  geom_mosaic(aes(x = product(y, marital), fill = y)) +
  mosaic_theme +
  xlab("Marital status") +
  ylab(NULL)

#education

fun_crosstable(bank_data, "education", "y")


bank_data = bank_data %>% 
  filter(education != "illiterate")

bank_data = bank_data %>% 
  mutate(education = recode(education, "unknown" = "university.degree"))

#default

fun_crosstable(bank_data, "default", "y")

bank_data = bank_data %>% 
  select(-default) 

fun_crosstable(bank_data, "loan", "y")

#Housing

fun_crosstable(bank_data, "housing", "y")


chisq.test(bank_data$housing, bank_data$y)


bank_data = bank_data %>% 
  select(-housing)

#loan
fun_crosstable(bank_data, "loan", "y")

chisq.test(bank_data$loan, bank_data$y)

bank_data = bank_data %>% 
  select(-loan)

#Contact

fun_crosstable(bank_data, "contact", "y")


#Month

month_recode = c("jan" = "(01)jan",
                 "feb" = "(02)feb",
                 "mar" = "(03)mar",
                 "apr" = "(04)apr",
                 "may" = "(05)may",
                 "jun" = "(06)jun",
                 "jul" = "(07)jul",
                 "aug" = "(08)aug",
                 "sep" = "(09)sep",
                 "oct" = "(10)oct",
                 "nov" = "(11)nov",
                 "dec" = "(12)dec")

bank_data = bank_data %>% 
  mutate(month = recode(month, !!!month_recode))

fun_crosstable(bank_data, "month", "y")


bank_data %>% 
  ggplot() +
  aes(x = month, y = ..count../nrow(bank_data), fill = y) +
  geom_bar() +
  ylab("relative frequency")


# Day of the week
day_recode = c("mon" = "(01)mon",
               "tue" = "(02)tue",
               "wed" = "(03)wed",
               "thu" = "(04)thu",
               "fri" = "(05)fri")

bank_data = bank_data %>% 
  mutate(day_of_week = recode(day_of_week, !!!day_recode))


fun_crosstable(bank_data, "day_of_week", "y")

#Duration

bank_data = bank_data %>% 
  select(-duration)

#campaign

bank_data %>% 
  ggplot() +
  aes(x = campaign) +
  geom_bar() +
  facet_grid(y ~ .,
             scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 50, 5))


bank_data = bank_data %>% 
  filter(campaign <= 10)


bank_data %>% 
  ggplot() +
  aes(x = campaign) +
  geom_bar() +
  facet_grid(y ~ .,
             scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 10, 1))

fun_crosstable(bank_data, "campaign", "y")


bank_data = bank_data %>% 
  mutate(campaign = as.character(campaign))

#Pdays

table(bank_data$pdays)


bank_data = bank_data %>% 
  mutate(pdays_dummy = if_else(pdays == 999, "0", "1")) %>% 
  select(-pdays)  

fun_crosstable(bank_data, "pdays_dummy", "y")

#previous

table(bank_data$previous)


bank_data = bank_data %>% 
  mutate(previous = if_else(previous >=  2, "2+", if_else(previous == 1, "1", "0")))

fun_crosstable(bank_data, "previous", "y")

#Poutcome

fun_crosstable(bank_data, "poutcome", "y")


#Bivariate analysis

bank_data %>% 
  select(emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed) %>% 
  cor() %>% 
  corrplot(method = "number",
           type = "upper",
           tl.cex = 0.8,
           tl.srt = 45,
           tl.col = "black")


gg_emp.var.rate = fun_gg_freq("emp.var.rate")
gg_cons.price.idx = fun_gg_freq("cons.price.idx")
gg_cons.conf.idx = fun_gg_freq("cons.conf.idx")
gg_euribor3m = fun_gg_freq("euribor3m")
gg_nr.employed = fun_gg_freq("nr.employed")

plot_grid(gg_emp.var.rate + theme(legend.position = "none") + ylab("Frequency"), 
          gg_cons.price.idx + theme(legend.position = "none"),
          gg_cons.conf.idx + theme(legend.position = "none"),
          gg_euribor3m + theme(legend.position = "none"),
          gg_nr.employed + theme(legend.position = "none"),
          get_legend(gg_cons.conf.idx),
          align = "vh")

bank_data = bank_data %>% 
  select(-emp.var.rate)

head(bank_data)

#checking for correlation
bank_data %>% 
  select(cons.price.idx, cons.conf.idx, euribor3m, nr.employed) %>% 
  cor() %>% 
  corrplot(method = "number",
           type = "full",
           tl.cex = 0.8,
           tl.srt = 45,
           tl.col = "black")


#Predictive models

#data preparation

bank_data$age = fun_reorder_levels(bank_data, "age", "low")
bank_data$job = fun_reorder_levels(bank_data, "job", "unemployed")
bank_data$marital = fun_reorder_levels(bank_data, "marital", "single")
bank_data$education = fun_reorder_levels(bank_data, "education", "basic.4y")
bank_data$contact = fun_reorder_levels(bank_data, "contact", "telephone")
bank_data$month = fun_reorder_levels(bank_data, "month", "(03)mar")
bank_data$day_of_week = fun_reorder_levels(bank_data, "day_of_week", "(01)mon")
bank_data$campaign = fun_reorder_levels(bank_data, "campaign", "1")
bank_data$previous = fun_reorder_levels(bank_data, "previous", "0")
bank_data$poutcome = fun_reorder_levels(bank_data, "poutcome", "nonexistent")
bank_data$pdays_dummy = fun_reorder_levels(bank_data, "pdays_dummy", "0")


glimpse(bank_data)


set.seed(1234)

ind = createDataPartition(bank_data$y,
                          times = 1,
                          p = 0.8,
                          list = F)
bank_train = bank_data[ind, ]
bank_test = bank_data[-ind, ]


#logistic regression

logistic = glm(y ~ .,
               data = bank_train,
               family = "binomial")

summary(logistic
        )

fun_imp_ggplot_split(logistic)

logistic_train_score = predict(logistic,
                               newdata = bank_train,
                               type = "response")


logistic_test_score = predict(logistic,
                              newdata = bank_test,
                              type = "response")


logistic_train_cut = 0.2
logistic_train_class = fun_cut_predict(logistic_train_score, logistic_train_cut)
# matrix
logistic_train_confm = confusionMatrix(logistic_train_class, bank_train$y, 
                                       positive = "1",
                                       mode = "everything")
logistic_train_confm



#Logistic regression 2 (simple)


logistic_2 = glm(y ~ . - job - marital - education - previous - euribor3m - cons.conf.idx - campaign,
                 data = bank_train,
                 family = "binomial")

summary(logistic_2)

logistic_train_score_2 = predict(logistic_2,
                                 newdata = bank_train,
                                 type = "response")

logistic_test_score_2 = predict(logistic_2,
                                newdata = bank_test,
                                type = "response")

#confusion matrix

logistic_train_cut_2 = 0.2
logistic_train_class_2 = fun_cut_predict(logistic_train_score_2, logistic_train_cut_2)
# matrix
logistic_train_confm_2 = confusionMatrix(logistic_train_class_2, bank_train$y, 
                                         positive = "1",
                                         mode = "everything")
logistic_train_confm_2


#Decision tree

modelLookup("rpart")


tune_grid = expand.grid(
  cp = seq(from = 0, to = 0.01, by = 0.001)
)

tune_control = trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds 
  summaryFunction = prSummary,
  verboseIter = FALSE, # no training log
  allowParallel = FALSE # FALSE for reproducible results 
)

rpart1_tune = train(
  y ~ .,
  data = bank_data,
  metric = "F",
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "rpart"
)


ggplot(rpart1_tune) +
  theme(legend.position = "bottom")

tree = rpart(y ~ .,
             data = bank_train,
             cp = rpart1_tune$bestTune)

#Decision tree plot
rpart.plot(tree)


tree_train_score = predict(tree,
                           newdata = bank_train,
                           type = "prob")[, 2]

tree_test_score = predict(tree,
                          newdata = bank_test,
                          type = "prob")[, 2]

#confusion matrix

tree_train_cut = 0.25
tree_train_class = fun_cut_predict(tree_train_score, tree_train_cut)
tree_train_confm = confusionMatrix(tree_train_class, bank_train$y, 
                                   positive = "1",
                                   mode = "everything")
tree_train_confm
