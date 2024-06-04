###########################################################################################################
# Statistical Analyses - Iguanas from Above
#
# Authors: Andres Camilo Marmol Guijarro & Andrea Varela
###########################################################################################################


#These analyses uses the filtered GS dataset with expert counts and aggregated volunteer counts with the median, mode and hdbscan clustering algorithm (obtained with Zooniverve_Clustering pipeline) as methods used to count marine iguanas in the images.

#volunteers counts were calculated on images selected using the 5-volunteer minimum threshold.
#partials are not included in these analyses.

# Clear environment

# Set up working directory 

# Load packages 
library(dplyr) # subset and transform data
library(ggplot2) # graphic package 
library(tidyr) # data reorganization
library(lme4) # generalized linear models analysis
library(emmeans) # posthoc test for generalized linear models using poisson correction family
library(pbkrtest)
library(Matrix)
library(performance)


#read the dataframe with all GS images, from the three phases together.
dfall = read.csv("iguanas-from-above-aggregateddataset-GoldStandard.csv", sep=";")
names(dfall)

#Extract all images WITH iguanas present (from the expert view)
dfall2 = dplyr::filter(dfall, expert_count > "0")

##Reorganize the dataframe for statistical analyses
dfall3 <- tidyr::gather(dfall2, key = "method", value = "counts",
                        expert_count, median_count, mode_count, hdbscan_count)

write.csv(dfall3, file="Iguanas all_stats_dataset-tidyr.csv")



############# STATISTICAL ANALYSES INVESTIGATING DIFFERENCES AMONG METHODS ##################


###1. Are there differences among the methods used to count marine iguanas?

##1.1. All Phases analyzed - Generalized linear model (Glm)
model1 = glm(counts~method, data=dfall3, family=quasipoisson) #adecuate family for counts data
summary(model1)

#1.2. ANOVA of the model
anova(model1, test="Chi")

#1.3. Posthoc test looking for differences between pairs of methods
emmall = emmeans(model1, ~method)
plot(emmall)
emmip(emmall, ~method, CIs = TRUE)
emmall
pairs(emmall)


## --------------------------------------------


###2. Are there differences among the methods used to count marine iguanas when adding the factor phase?

##2.1. All Phases analyzed - Generalized linear model (Glm)

#2.1.1. Interaction model
model2_1 = glm(counts~method*phase, data=dfall3, family=quasipoisson) #adecuate family for counts data
summary(model2_1)

#2.1.2  Principal effect model (simpler)
model2_2 = glm(counts~method+phase , data=dfall3, family=quasipoisson)
summary(model2_2)

#2.2. ANOVA between models
anova(model2_1, model2_2, test = "Chi") # If Non-different -> use simpler model

#2.3. ANOVA of the chosen model
anova(model2_2, test="Chi")

#2.4. Posthoc test looking for differences between pairs of methods
#among methods+phases
emm = emmeans(model2_2, ~method+phase)
plot(emm)
emmip(emm, method~phase, CIs = TRUE)
emm
pairs(emm)

#among phases
emm = emmeans(model2_2, ~phase)
plot(emm)
emmip(emm, ~phase, CIs = TRUE)
emm
pairs(emm)


## --------------------------------------------


###3. Are there differences among the methods used to count marine iguanas when analyzing each phase independently?

##Create new dataframes per phase
dff1 = dplyr::filter(dfall3, phase == "1st") 
dff2 = dplyr::filter(dfall3, phase == "2nd")
dff3 = dplyr::filter(dfall3, phase == "3rd")

### 1st phase
names(dff1)
data.frame(table(dff1$subject_id))

##3.1. Generalized linear model (Glm)
model3 <- glm(counts ~ method, data = dff1, family = quasipoisson) #adecuate family for counts data
summary(model3)

##3.2. ANOVA of the model
anova(model3, test="Chi")
  
##3.3. Posthoc test looking for differences between pairs of methods
emm1 = emmeans(model3, ~method)
plot(emm1)
emmip(emm1, ~method, CIs = TRUE)
emm1
pairs(emm1)

## --- Repeat for phase 2 and 3 ----------



############# STATISTICAL ANALYSES INVESTIGATING DIFFERENCES AMONG METHODS DEPENDING ON THE IMAGE QUALITY ##################

##4. Are there differences among the methods used to count marine iguanas whenadding the factor quality?

##4.1. All phases analyzed - Generalized linear model (Glm)

#4.1.1. Interaction model
model4_1 <- glm(counts ~ method*quality, data=dfall3, family=quasipoisson)
summary(model4_1)

#4.1.2. Principal effects model (simpler)
model4_2 <- glm(counts ~ method+quality, data=dfall3, family=quasipoisson)
summary(model4_2)

##4.2. ANOVA between models
anova(model4_1, model4_2, test="Chi") # If Non-differences, use the simpler model

##4.3. ANOVA of the chosen model
anova(model4_2, test="Chi")

##4.4. Posthoc test looking for differences between pairs of methods
#among methods+quality
emmallmq = emmeans(model4_2, ~method+quality)
plot(emmallmq)
emmip(emmallmq, method~quality, CIs = TRUE)
emmallmq
pairs(emmallmq)

#between quality
emmallq = emmeans(mode4_2, ~quality)
plot(emmallq)
emmip(emmallq, ~quality, CIs = TRUE)
emmallq
pairs(emmallq)


## --------------------------------------------


##5. Are there differences among the methods used to count marine iguanas when adding the factor quality and analyzing each phase independently?

###1st Phase

##5.1. Interaction model
model5_1 = glm(counts~method*quality, data = dff1, family = quasipoisson)
summary(model5_1)

##5.2. Principal effects model (simpler)
model5_2 = glm(counts~method+quality, data = dff1, family = quasipoisson)
summary(model5_2)

##5.3. ANOVA between models
anova(model5_1, model5_2, test="Chi") #If Non-different, use simpler model
  
#5.4. ANOVA of the model chosen
anova(model5_2, test="Chi")

##5.5. Posthoc test looking for differences between pairs of methods
#among methods+quality
emm1mq = emmeans(model5_2, ~method+quality)
plot(emm1mq)
emmip(emm1mq, method~quality, CIs = TRUE)
emm1mq
pairs(emm1mq)

#between quality
emm1q = emmeans(model5_2, ~quality)
plot(emm1q)
emmip(emm1q, ~quality, CIs = TRUE)
emm1q
pairs(emm1q)


## --- Repeat for phase 2 and 3 ----------



############# STATISTICAL ANALYSES INVESTIGATING DIFFERENCES AMONG METHODS DEPENDING ON THE NUMBER OF IGUANAS IN THE IMAGE ##################


#read the dataframe with all GS images, from the three phases together.
dfall = read.csv("iguanas-from-above-aggregateddataset-GoldStandard.csv", sep=";")
names(dfall)

#Subset images based on these categories: from 0 to 5 iguanas, from 6 to 10 iguanas and more than 10
dfall0_5 <- dplyr::filter(dfall, expert_count <= 5)
dfall6_inf <- dplyr::filter(dfall, expert_count > 5)
dfall6_10 <- dplyr::filter(dfall6_inf, expert_count <= 10)
dfall10_inf <- dplyr::filter(dfall6_inf, expert_count > 10)

#Create qualitative variable of quantity
dfall0_5$quantity = "a 0-5"
dfall6_10$quantity = "b 6-10"
dfall11_inf$quantity = "c 11-inf"

#Binding subset data table
dfall_1 = dplyr::bind_rows(dfall0_5, dfall6_10)
dfall_1 = dplyr::bind_rows(dfall_1, dfall11_inf)

#Reorganize the dataframe for statistical analyses
dfall_2 = tidyr::gather(dfall_1, key = "method", value = "counts",
                        expert_count, median_count, mode_count, hdbscan_count)


write.csv(dfall_2, file="Iguanas all_stats_dataset-quantity-tidyr.csv")


## --------------------------------------------


####6. Are there differences among the methods used to count marine iguanas when adding the factor quantity of iguanas?

##6.1. All phases analyzed - Generalized linear models (Glm)

#6.1.1. Interaction model
model6_1 = glm(counts ~ method * quantity, data=dfall_2, family=quasipoisson)
summary(model6_1)

#6.1.2. Principal effects models (simpler)
model6_2 = glm(counts ~ method + quantity, data=dfall_2, family=quasipoisson)
summary(model6_2)

#6.2. ANOVA between models
anova(model6_1, model6_2, test="Chi") # If Non-different, use the simpler model

##6.3. ANOVA of the chosen model
anova(model6_2, test="Chi")

##6.4. Posthoc test looking for differences between pairs of methods
emm_all = emmeans(model6_2, ~method+quantity)
plot(emm_all)
emmip(emm_all, method~quantity, CIs = TRUE)
emm_all
pairs(emm_all)


## --------------------------------------------


####7. Are there differences among the methods used to count marine iguanas adding when adding the factor quantity of iguanas and analyzing each phase independently?

##Create new dataframes per phase
dfall_1f <- dplyr::filter(dfall_2, phase == "1st")
dfall_2f <- dplyr::filter(dfall_2, phase == "2nd")
dfall_3f <- dplyr::filter(dfall_2, phase == "3rd")


### 1st phase
names(dfall_1f)
data.frame(table(dfall_1f$subject_id))


##7.1. Interaction model
model7_1 <- glm(counts ~ method*quantity, data = dfall_1f, family=quasipoisson)
summary(model7_1)

##7.2. Principal effects model (simpler)
model7_2 <- glm(counts ~ method+quantity, data = dfall_1f, family=quasipoisson)
summary(model7_2)

##7.3. ANOVA between models
anova(model7_1, model7_2, test="Chi") # If Non-different, use simpler model
  
##7.4. ANOVA of the chosen model
anova(model7_2, test="Chi")

##7.5. Posthoc test between pairs
emm_1f_2 = emmeans(model7_2, ~method+quantity)
plot(emm_1f_2)
emmip(emm_1f_2, method~quantity, CIs = TRUE)
emm_1f_2
pairs(emm_1f_2)


## --- Repeat for phase 2 and 3 ----------



############# STATISTICAL ANALYSES TO EXPLORE THE AGGREGATING METHOD WITH THE BEST FIT (R-squared values) TO THE EXPERT ##################


dfcompare = dfall3 %>%
  arrange(subject_id, method) %>%
  group_by(method) %>%
  mutate(id=row_number()) #%>%

dfexpert = dfcompare %>%
  filter(method == "expert_count")

dfmedian = dfcompare %>%
  filter(method == "median_count")

dfmode = dfcompare %>%
  filter(method == "mode_count")

dfhdbscan = dfcompare %>%
  filter(method == "hdbscan_count")

dfcompall = dfexpert

dfcompall$method_median = dfmedian$counts
dfcompall$method_mode = dfmode$counts
dfcompall$method_hdbscan = dfhdbscan$counts


##R-square for each model with graph

#median
compmedian = glm(counts~method_median, data = dfcompall, family=quasipoisson)
summary(compmedian)

ggplot(dfcompall, aes(x=counts, y=method_median))+
  geom_point(stat = "identity")+
  stat_smooth(aes(), method = "glm", formula = y ~ x, method.args = list(family = "quasipoisson"), size = 1)

#mode
compmode = glm(counts~method_mode, data = dfcompall, family=quasipoisson)
summary(compmode)

ggplot(dfcompall, aes(x=counts, y=method_mode))+
  geom_point(stat = "identity")+
  stat_smooth(aes(), method = "glm", formula = y ~ x, method.args = list(family = "quasipoisson"), size = 1)

#hdbscan
comphdbscan = glm(counts~method_hdbscan, data = dfcompall, family=quasipoisson)
summary(comphdbscan)

ggplot(dfcompall, aes(x=counts, y=method_hdbscan))+
  geom_point(stat = "identity")+
  stat_smooth(aes(), method = "glm", formula = y ~ x, method.args = list(family = "quasipoisson"), size = 1)

##Compare models
compare_performance(compmedian,compmode,comphdbscan, rank=TRUE, verbose=TRUE)

compare_performance(compmedian,compmode,comphdbscan, verbose=TRUE)

# information about the poisson link function formula can be found in the following link
# https://towardsdatascience.com/generalized-linear-models-9cbf848bb8ab#:~:text=Link%20function%20literally%20%E2%80%9Clinks%E2%80%9D%20the,be%20positive%20(explained%20later).


#Graph with lines of all metrics R-square analysis
dfcompall_2 = dfcompall %>%
  pivot_longer(cols = c(method_mode, method_median, method_hdbscan), names_to = "metric_type", values_to = "metric_count")

# graph

plot_rsq = ggplot(dfcompall_2, aes(x=counts, y=metric_count))+
  geom_point(aes(fill=metric_type), shape=21,position=position_dodge2(width=0.6))+
  stat_smooth(aes(colour=metric_type), method = "glm", formula = y ~ x, method.args = list(family = "quasipoisson"), size = 1, alpha=0.2)+
  
ylab(expression(paste("Iguanas counts"))) +
xlab(expression(paste("Expert counts")))+
  
  
theme(legend.background = element_rect(size=0.5),
        plot.background = element_blank(),
        panel.background = element_blank(),
        axis.line=element_line(color="black"),
        )

plot_rsq
