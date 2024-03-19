#### LIBRERIAS ------
library(openxlsx) # para importar desde excel
library(tidyverse) # manipulacion de datos
library(ggplot2) # graficos
library(magrittr) # %>%
library(lubridate) # Manipulacion de fechas
library(stringr) # Manipulacion de texto
library(skimr) # Descriptivas
library(tidymodels) # Facilita modelamiento - new way
library(glmnet) # Regularizacion L2/L1/elastic net models\
library(glmnetUtils)
library(ggforce) # graficos ggplot
library(doParallel) # Computacion en paralelo
library(kknn) # KNN

## IMPORTAR Y ENTENDER -------------

data_banco <- read_delim("Data/bank-additional-full.csv", delim= ";")
glimpse(data_banco)


# Convertir a factor
data_banco %<>% 
  mutate( y = factor(y, 
                     levels= c("yes","no"), 
                     labels= c("si", "no")) 
  ) -> data_banco
# Verificar
# str : ver la estructura de la variable
data_banco %$% str(y)


##  Explorar el balanceo -----------------------
data_banco %>%
group_by(y) %>%
  summarise( Frec= n()) %>%
  mutate(Prop= Frec/ sum(Frec) ) 


# Convertir a factor
data_banco %>% mutate( 
  education = factor( education, 
                      levels= c("illiterate", "basic.4y", "basic.6y", 
                                "basic.9y", "high.school",  "professional.course",
                                "university.degree", "unknown" ), 
                      labels= c("No_Educ", "4A_Bas", "6A_Bas",
                                "9A_Bas", "Bachill", "Tecnico",
                                "Univer", "Descon") )  
) -> data_banco


# Convertir a factor
data_banco %>% mutate( 
  month = factor( month, 
                  levels= c("mar", "apr", "may", "jun", "jul", 
                            "aug", "sep", "oct", "nov", "dec" ), 
                  labels= c("Mar", "Abr", "May", "Jun", "Jul", 
                            "Ago", "Sep", "Oct", "Nov", "Dec" )),
  day_of_week = factor( day_of_week, 
                        levels= c("mon", "tue", "wed", "thu", "fri" ), 
                        labels= c("Lun", "Mar", "Mie", "Jue", "Vie" )),
) -> data_banco



## + Variables con muchas categorias --------------
data_banco %>%
  group_by(job) %>% 
  summarise(Frec= n()) %>% 
  arrange( -Frec) %>% 
  mutate( 
    Porcentaje= round(Frec/ sum(Frec), 4) *100,
    PorcentajeAcum= round( cumsum(Frec)/ sum(Frec), 4) *100
  )


ggplot(data_banco, aes(y, cons.price.idx)) + 
  geom_boxplot(aes(fill = y)) +
  labs(title = "Boxplot Indice consumidor",
       x= "Adquiere prestamo") -> plot_consprice
ggplot(data_banco, aes(y, cons.conf.idx)) + 
  geom_boxplot(aes(fill = y)) +
  labs(title = "Boxplot Confianza consumidor",
       x= "Adquiere prestamo") -> plot_consconf
ggplot(data_banco, aes(y, euribor3m)) + 
  geom_boxplot(aes(fill = y)) +
  labs(title = "Boxplot tasa euribor",
       x= "Adquiere prestamo") -> plot_euribor3m
ggplot(data_banco, aes(y, nr.employed)) + 
  geom_boxplot(aes(fill = y)) +
  labs(title = "Boxplot Numero empleados",
       x= "Adquiere prestamo") -> plot_employed
cowplot::plot_grid	(plot_consprice, plot_consconf, plot_euribor3m, 
          plot_employed, nrow = 2, ncol= 2)


data_banco %>% 
  select_if( is.numeric) %>% 
  cor %>% 
  corrplot::corrplot(method = 'number', order = 'hclust')


## MODELAMIENTO ------------------
## + Partición Train-Test 

set.seed(1234) # Semilla para aleatorios
bco_split <- data_banco %>%
  initial_split(prop = 0.8,
                strata = y)
train <- training(bco_split)
dim(train)

test <- testing(bco_split)
dim(test)



## + Preprocesamiento con recipe ----------

rct_bcoPor <- train %>% recipe(y ~ . ) %>%
  step_rm(pdays, duration) %>% # Eliminar Satisfaccion 
  step_normalize( all_numeric(), -all_outcomes()) %>% # Normalizacion
  step_other(all_nominal(), -all_outcomes() ) %>% 
  step_dummy(all_nominal(), -all_outcomes() ) %>% # Dummy
  # step_corr(all_numeric(), -all_outcomes(), threshold = 0.9) %>%
  # Se elimina manualmente para poder agregar las interacciones
  step_rm(nr.employed, emp.var.rate) %>%
  step_nzv(all_predictors()) %>% 
  themis::step_upsample(y, over_ratio = 0.9, skip= TRUE, seed= 123) %>% 
  step_sample(size = 5000, skip = TRUE, seed= 456 )


## + Estrategia de remuestreo -----------------

set.seed(1234)
cv_banco <- vfold_cv(train, v = 5, repeats = 1, strata = y)
cv_banco


## + Métricas --------

metricas <- metric_set(accuracy, sens, spec, bal_accuracy)
metricas


## + Modelamiento con KNN -------------------
## ++ Especificacion del modelo ----------------

knn_sp <- 
  nearest_neighbor(neighbors= tune(), weight_func= tune(), dist_power= tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

knn_sp %>% 
  translate()

## ++ Workflow ----------------
knn_wflow <- 
  workflow() %>% 
  add_recipe(rct_bcoPor) %>% 
  add_model(knn_sp)
knn_wflow

## ++ Paralelización -------------
parallel::detectCores(logical=FALSE)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
# parallel::stopCluster(cl)


## ++ Afinamiento de Hiperparametros ----------------------
## ++++ Malla de busqueda -------------
set.seed(123)
knn_grid <- knn_sp %>%
  parameters() %>% 
  update(
    weight_func= weight_func(c("biweight", "rectangular")),
    neighbors= neighbors( range = c(7, 17))
    ) %>% 
  grid_latin_hypercube(size = 6)

knn_grid



## ++++ Entrenamiento de la malla de busqueda --------------
set.seed(123)
knn_tuned <- tune_grid(
  knn_wflow,
  resamples= cv_banco,
  grid = knn_grid,
  metrics = metricas,
  control= control_grid(allow_par = T)
)
knn_tuned


show_best(knn_tuned, metric = 'accuracy', n = 5)
show_best(knn_tuned, metric = 'sens', n = 5)
show_best(knn_tuned, metric = 'spec', n = 5)
show_best(knn_tuned, metric = 'bal_accuracy', n = 5)

## ++ Modelo final ------------
knn_pars_fin <- select_best(knn_tuned, metric = 'bal_accuracy')
knn_wflow_fin <- 
  knn_wflow %>% 
  finalize_workflow(knn_pars_fin)
knn_fitted <- fit(knn_wflow_fin, train)
knn_fitted


## ++ Metricas en el train ------------
train %>% 
  predict(knn_fitted, new_data = . ) %>% 
  mutate(Real= train$y) %>% 
  conf_mat(truth = Real, estimate = .pred_class ) %>% 
  summary


## ++ Evaluamos en el test ------------------

test %>% 
  predict(knn_fitted, new_data = . ) %>% 
  mutate(Real= test$y) %>% 
  conf_mat(truth = Real, estimate = .pred_class ) %>% 
  summary
