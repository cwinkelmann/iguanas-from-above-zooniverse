###########################################################################################################
# Presence/Absence Analyses - Iguanas from Above
#
# Author: Andrea Varela-Jaramillo
###########################################################################################################


##This analysis uses the filtered Gold Standard (GS) dataset with the 20/30 Yes and No answers aggregated into 1 answer per image (obtained with the Panoptes_Data_Prep pipeline) and the expert answers, for comparisons.


###1. PRESENCE/ABSENCE: compare yes and no answers using a majority vote or most frequent answer criteria (Swanson et al. 2015).
#if equal, print Y as we have seen volunteers tend to miss iguanas instead of overcount.
#expert answers were added in the column named presence_absence_exp.
#example for phase 3

GSall <- read.csv("1-TO-GS-comparison.csv", sep = ";") #read the dataset
names(GSall)

GSall$presence_absence_vol <- ifelse(GSall$presence_yes > GSall$presence_no, 'Y', ifelse(GSall$presence_yes < GSall$presence_no, 'N', 'Y')) #print the selected answer for presence/absence in a new column
unique(GSall$presence_absence_vol)
data.frame(table(GSall$presence_absence_vol)) #gets the number of images with yes and no from volunteers answers
data.frame(table(GSall$presence_absence_exp)) #gets the number of images with yes and no from expert answers


##Compare volunteers answers against expert answers.
GSall$comparison <- ifelse(GSall$presence_absence_exp == GSall$presence_absence_vol, 'Correct', 'Incorrect') #print the selected answer in a new column
unique(GSall$comparison)
data.frame(table(GSall$comparison)) #gets the number of images with correct and incorrect answers

##Calculate percents from results obtained (example).
(1094*100)/1156 #correct
#R: 95
(62*100)/1156 #incorrect
#R: 5

#export your results
write.csv(GSall, file="3-GS-results_mv.csv")


## --------------------------------------------


##1.1.Same analysis for GS images WITH iguanas present (from the expert view).
GSY <- read.csv("3-T0-GS-comparison.csv", sep = ";")

##Subset images with iguanas.
GSY <- subset(GSY, presence_absence_exp == "Y")
unique(GSY$presence_absence_exp)

GSY$presence_absence_vol <- ifelse(GSY$presence_yes > GSY$presence_no, 'Y', ifelse(GSY$presence_yes < GSY$presence_no, 'N', 'Y')) #print the selected answer for presence/absence in a new column
unique(GSY$presence_absence_vol)
data.frame(table(GSY$presence_absence_vol))
data.frame(table(GSY$presence_absence_exp))


##Compare volunteers answers against expert answers.
GSY$comparison <- ifelse(GSY$presence_absence_exp == GSY$presence_absence_vol, 'Correct', 'Incorrect') #print the selected answer in a new column
unique(GSY$comparison)
data.frame(table(GSY$comparison))

#export your results
write.csv(GSY, file="3-GS-results_mv_Y.csv")


## --------------------------------------------


##1.2. Same analysis for GS images WITHOUT iguanas present (from the expert view).
GSN <- read.csv("3-T0-GS-comparison.csv", sep = ";")

##Subset images with no iguanas.
GSN <- subset(GSN, presence_absence_exp == "N")
unique(GSN$presence_absence_exp)

GSN$presence_absence_vol <- ifelse(GSN$presence_yes > GSN$presence_no, 'Y', ifelse(GSN$presence_yes < GSN$presence_no, 'N', 'Y')) #print the selected answer for presence/absence in a new column
unique(GSN$presence_absence_vol)
data.frame(table(GSN$presence_absence_vol))
data.frame(table(GSN$presence_absence_exp))

##Compare volunteers answers against expert answers.
GSN$comparison <- ifelse(GSN$presence_absence_exp == GSN$presence_absence_vol, 'Correct', 'Incorrect') #print the selected answer in a new column
unique(GSN$comparison)
data.frame(table(GSN$comparison))

#export your results
write.csv(GSN, file="3-GS-results_mv_N.csv.csv")


## --------------------------------------------


###2. PRESENCE/ABSENCE: compare yes and no answers looking for the minimum number of volunteers needed for correct identification and accuracy improvement.
#This first example accepts as correct for iguana presence when at least 1 volunteer from the 20 selected yes.
#Repeat the analysis from 2 to 11 volunteers.
GS1 <- read.csv("GS-comparison.csv", sep = ";")

GS1$presence_absence_vol <- ifelse(GS1$presence_yes > 0, 'Y', 'N') #print the selected answer for presence/absence in a new column
unique(GS1$presence_absence_vol)
data.frame(table(GS1$presence_absence_vol))
data.frame(table(GS1$presence_absence_exp))

##Compare volunteers answers against expert answers.
GS1$comparison <- ifelse(GS1$presence_absence_exp == GS1$presence_absence_vol, 'Correct', 'Incorrect') #print the selected answer in a new column
unique(GS1$comparison)
data.frame(table(GS1$comparison))

##Calculate percents from results (example).
(1113*100)/1156
#R: 56.7
(43*100)/1156
#R: 43.3

#Create data.frame for the 11 results (example).
dfvol <- data.frame(Criteria = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"), Correct = c(56.7, 86.6, 94.8, 96.3, 96.5, 96.5, 96.5, 96.5, 96.3, 95.8, 95.6), Incorrect =c(43.3, 13.4, 5.2, 3.7, 3.5, 3.5, 3.5, 3.5, 3.7, 4.2, 4.4))
dfvol
#R: 5 volunteers identified as the minimun threshold to obtain the highest percentages of accuracy between the volunteers and the experts for iguana presence/absence analysis in our GS dataset.

#plot a line graph to show the tendency and the best accuracy found.
plot(dfvol$Correct, type="b", col="#629E0D", pch=16, cex =2, xlim=c(1,17), ylim=c(50, 100), axes=FALSE, main = "Minimum threshold analysis GS images", xlab = "Number of volunteers selecting YES for iguana presence", ylab = "Percentage of accuracy compared to the experts")
axis(1, at =1:11)
axis(2)

#export your results
write.csv(GS5, file="3-GS-results_5th.csv")


## --------------------------------------------


##2.1. Same analysis for GS images WITH iguanas present (from the expert view) using the 5-minimum threshold identified.
GS <- read.csv("3-TO-GS-comparison.csv", sep = ";")

##Subset images with iguanas.
GSY5 <- subset(GS, presence_absence_exp == "Y")

GSY5$presence_absence_vol <- ifelse(GSY1$presence_yes > 4, 'Y', 'N') #print the selected answer for presence/absence in a new column
unique(GSY5$presence_absence_vol)
data.frame(table(GSY5$presence_absence_vol))
data.frame(table(GSY5$presence_absence_exp))

##Compare volunteers answers against expert answers.
GSY5$comparison <- ifelse(GSY5$presence_absence_exp == GSY5$presence_absence_vol, 'Correct', 'Incorrect') #print the selected answer in a new column
unique(GSY5$comparison)
data.frame(table(GSY5$comparison))

#export your results.
write.csv(GSY5, file="3-GS-results_5th_Y.csv")


## --------------------------------------------


##2.2. Same analysis for GS images WITHOUT iguanas present (from the expert view) using the 5 minimum threshold identified.
GS <- read.csv("3-TO-GS-comparison.csv", sep = ";")

##Subset images with no iguanas.
GSN5 <- subset(GS, presence_absence_exp == "N")

GSN5$presence_absence_vol <- ifelse(GSN5$presence_yes > 4, 'Y', 'N') #print the selected answer for presence/absence in a new column
unique(GSN5$presence_absence_vol)
data.frame(table(GSN5$presence_absence_vol))
data.frame(table(GSN5$presence_absence_exp))

##Compare volunteers answers against expert answers.
GSN5$comparison <- ifelse(GSN5$presence_absence_exp == GSN5$presence_absence_vol, 'Correct', 'Incorrect') #print the selected answer in a new column
unique(GSN5$comparison)
data.frame(table(GSN5$comparison))

#export your results.
write.csv(GSN5, file="3-GS-results_5th_N.csv")
