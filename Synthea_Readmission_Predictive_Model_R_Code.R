#############################################################
########          POC R Studio on MS Azure           ########
########          Readmission Predictive Model       ######## 
########                August 2022                  ######## 
#############################################################

#Access Azure Portal - Storage Account and Read files
#Data Wrangling and Predictive Model
#Step 1 - Include Azure profile
source("C:/dev/Rprofile.R")
#Step 2 - Invoke necessary libraries for analyses and modeling.
library(AzureStor)    #Manage storage in Microsoft's 'Azure' cloud
library(AzureRMR)     #Interface to 'Azure Resource Manager'
library(psych)        #A general purpose toolbox for personality, psychometric theory and experimental psychology. Functions are primarily for multivariate analysis. 
library(ggplot2) 	    #A system for creating graphics, based on "The Grammar of Graphics". 
library(caret) 		    #Misc functions for training and plotting classification and regression models.
library(rpart) 		    #Recursive partitioning for classification, regression and survival trees.  
library(rpart.plot) 	#Plot 'rpart' models. Extends plot.rpart() and text.rpart() in the 'rpart' package.
library(RColorBrewer) #Provides color schemes for maps (and other graphics). 
library(party)		    #A computational toolbox for recursive partitioning.
library(partykit)	    #A toolkit with infrastructure for representing, summarizing, and visualizing tree-structure.
library(pROC) 		    #Display and Analyze ROC Curves.
library(ISLR)		      #Collection of data-sets used in the book 'An Introduction to Statistical Learning with Applications in R.
library(randomForest)	#Classification and regression based on a forest of trees using random inputs.
library(dplyr)		    #A fast, consistent tool for working with data frame like objects, both in memory and out of memory.
library(ggraph)		    #The grammar of graphics as implemented in ggplot2 is a poor fit for graph and network visualizations.
library(igraph)		    #Routines for simple graphs and network analysis.
library(mlbench) 	    #A collection of artificial and real-world machine learning benchmark problems, including, e.g., several data sets from the UCI repository.
library(GMDH2)		    #Binary Classification via GMDH-Type Neural Network Algorithms.
library(apex)		      #Toolkit for the analysis of multiple gene data. Apex implements the new S4 classes 'multidna'.
library(mda)		      #Mixture and flexible discriminant analysis, multivariate adaptive regression splines.
library(WMDB)		      #Distance discriminant analysis method is one of classification methods according to multiindex.
library(klaR)		      #Miscellaneous functions for classification and visualization, e.g. regularized discriminant analysis, sknn() kernel-density naive Bayes...
library(kernlab)	    #Kernel-based machine learning methods for classification, regression, clustering, novelty detection.
library(readxl)    	  #n Import excel files into R. Supports '.xls' via the embedded 'libxls' C library.                                                                                                                                                                 
library(GGally)  	    #The R package 'ggplot2' is a plotting system based on the grammar of graphics.                                                                                                                                                                  
library(mctest)		    #Package computes popular and widely used multicollinearity diagnostic measures.
library(sqldf)		    #SQL for dataframe wrangling.
library(reshape2)     #Pivoting table
library(anytime)      #Caches TZ in local env
library(survey)       #Summary statistics, two-sample tests, rank tests, glm.... 
library(mice)         #Library for multiple imputation
library(MASS)         #Functions and datasets to support Venables and Ripley
library(rjson)        #Load the package required to read JSON files.
library(RISmed)       #RISmed is a portmanteau of RIS (for Research Information Systems, a common tag format for bibliographic data) and PubMed.


#Apply credentials from profile
az <- create_azure_login(tenant=Azure_tenantID)

# same as above
blob_endp <- blob_endpoint("https://olastorageac.blob.core.windows.net/",key=Azure_Storage_Key)
file_endp <- file_endpoint("https://olastorageac.file.core.windows.net/",key=Azure_Storage_Key)

#An existing container
readmission_data <- blob_container(blob_endp, "readmissionmodel")

# list blobs inside a blob container
list_blobs(readmission_data)

#Temp download of files needed for data wrangling
storage_download(readmission_data, "readmission.csv", "~/readmission.csv")

#Read csv in memory
readm_data<-read.csv("readmission.csv")

#Delete Temp downloaded of files
file.remove("readmission.csv")

#Data cleaning and wrangling to get features required for modeling
#Number of columns
ncol(readm_data)
#Number of rows
nrow(readm_data)
#View fields in files
names(readm_data)
#View subset of file
head(readm_data,2)
#View structure of file
str(readm_data)

#Step 3
#Descriptive Statistics 
summary(readm_data)
#Check for missingness
sapply(readm_data, function(x) sum(is.na(x)))
readm_data <- readm_data[complete.cases(readm_data), ]

#Step 3A & 3B
#A. Preliminaries - Tests and Analysis
#NORMALITY TEST
attach(readm_data)
#shapiro.test(STRENGTH_NUM) # shapiroTest - Shapiro-Wilk Test
#Ideally, Multivariate Normality Testing would have been suitable - Research in This area -WIP.
qqnorm(Systolic_BP_mmHg)
qqline(Systolic_BP_mmHg)
#The null hypothesis for this test is that the data are normally distributed. The Prob < W value listed in #the #output is the p-value. If the chosen alpha level is 0.05 and the p-value is less than 0.05, then the #null #hypothesis that the data are normally distributed is rejected. If the p-value is greater than 0.05, #then the #null hypothesis has not been rejected.
#shapiro.test(Systolic_BP_mmHg)

#B
#Multicollinearity conducted via Variance 
#Inflation Factor (VIF), validated for significance using Akaike #Information Criteria (AIC), Bayesian #Information Criteria (BIC) and excluded some explanatory #variables.
#Sample Graphics                                                                                                                                                                                  
X<-subset (readm_data, select=c(Diastolic_BP_mmHg,                 
                                Systolic_BP_mmHg,
                                BodyWeight_kg, 
                                Heartrate_permin,
                                Respiratoryrate_permin,
                                BMI_kgperm2,
                                Glucose_mgperdL,
                                UreaNitrogen_mgperdL,
                                Creatinine_mgperdL,
                                Calcium_mgperdL,
                                Sodium_mmolperL                         
))
ggpairs(X)


#Relevant Dataset
#Readmission rate was calculated using 30 day all-cause logic, excluding cancer and ESRD patients
readm_data_indicator<-readm_data[,c(7:27)]


#Step 4
#Modeling
#Setting the random seed for replication
set.seed(89)
df<-readm_data_indicator

#Step 5
#setting up cross-validation
cv_control <- trainControl(method="repeatedcv", number = 10, allowParallel=TRUE)

#Step 6
#random sample half the rows 
halfsample = sample(dim(df)[1], dim(df)[1]/2) # half of sample
#create training and test data sets
df_train = df[halfsample, ]
df_test = df[-halfsample, ]


####################
###Modeling Steps### 
####################                                      
#Multicollinearity Tests not conducted since most of these methods do not need formal distributional assumptions
#Ensemble Methods - Single Tree, Bagging, Random Forests, Boosting, Logistic, Neural Network, Naive Bayes, Support Vector Machines and Nearest Neighbor

#1. #Simple/Single Classification Tree using method="ctree"
#Assumptions
#a.No formal distributional assumptions, random forests are non-parametric and can thus handle skewed and multi-modal data.
trainingmodeltree <- train(as.factor(READMISSION_INDICATOR) ~., data=df_train, method="ctree",trControl=cv_control, tuneLength = 10)
trainingmodeltree 
plot(trainingmodeltree)
trainingmodeltree 

#plot tree
plot(trainingmodeltree$finalModel, main="Regression Tree for Readmission Model")
#Get predicted probabilites for test dataset
probstree=predict(trainingmodeltree, newdata=df_test, type="prob")
head(probstree)

#Get class predictions for test dataset
classtesttree <-  predict(trainingmodeltree, newdata = df_test, type="raw")
head(classtesttree)

#Compute ROC 
roctree <- roc(df_test$READMISSION_INDICATOR,probstree[,"Yes"])
roctree 

#The ROC curve
plot(roctree,col=c(1))
#Compute area under curve (Closer to 1 is preferred)
auc(roctree)
#confusionmatrix
confusionmatrixtree<-confusion(predict(trainingmodeltree, df_train), df_train$READMISSION_INDICATOR)
confusionmatrixtree

# Overall Misclassification Rate
errorratetree<-(1-sum(diag(confusionmatrixtree))/sum(confusionmatrixtree))
errorratetree
#Sensitivity and Specificity
# Sensitivity - aka true positive rate, the recall, or probability of detection
sensitivitytree<-sensitivity(confusionmatrixtree)
sensitivitytree
## Specificity - aka true negative rate
specificitytree<-specificity(confusionmatrixtree)
specificitytree
#prediction
predtree<-data.frame(classtesttree)
predtree <- cbind(df_test, predtree)
#Write output of prediction to csv 
write.csv(predtree, file = "Prediction_tree.csv")

#2. #Bagging Model using method="treebag"
#Assumptions
#a.No formal distributional assumptions, random forests are non-parametric and can thus handle skewed and multi-modal data.
trainingmodelbagg <- train(as.factor(READMISSION_INDICATOR) ~ ., data=df_train, method="treebag", trControl=cv_control, importance=TRUE)
trainingmodelbagg

#Variable of Importance
plot(varImp(trainingmodelbagg))

#Get class predictions for training dataset
classtrainbagg <-  predict(trainingmodelbagg,  type="raw")
head(classtrainbagg)
#Get class predictions for test dataset
classtestbagg <-  predict(trainingmodelbagg, newdata = df_test, type="raw")
head(classtestbagg)
#Derive predicted probabilites for test dataset
probsbagg=predict(trainingmodelbagg, newdata=df_test, type="prob")
head(probsbagg)
#Compute ROC 
rocbagg <- roc(df_test$READMISSION_INDICATOR,probsbagg[,"Yes"])
rocbagg
#The ROC curve
plot(rocbagg,col=c(2))
##Compute area under curve (Closer to 1 is preferred)
auc(rocbagg)

#confusionmatrix
confusionmatrixbagg<-confusion(predict(trainingmodelbagg, df_train), df_train$READMISSION_INDICATOR)
confusionmatrixbagg
# Overall Misclassification Rate
errorratebagg<-(1-sum(diag(confusionmatrixbagg))/sum(confusionmatrixbagg))
errorratebagg
#Sensitivity and Specificity
# Sensitivity - aka true positive rate, the recall, or probability of detection
sensitivitybagg<-sensitivity(confusionmatrixbagg)
sensitivitybagg
## Specificity - aka true negative rate
specificitybagg<-specificity(confusionmatrixbagg)
specificitybagg
#prediction
predbagg<-data.frame(classtestbagg)
predbagg <- cbind(df_test, predbagg)
#Write output of prediction to csv 
write.csv(predbagg, file = "Prediction_bagg.csv")

#3. #Random Forest for Classification Trees using method="rf"
#Assumptions
#a.No formal distributional assumptions, random forests are non-parametric and can thus handle skewed and multi-modal data.
#NOTE:- This particular algorithm may take a long time to run, may need to recalibrate using smaller samples
trainingmodelranfor <- train(as.factor(READMISSION_INDICATOR) ~ ., data=df_train,method="rf",trControl=cv_control, importance=TRUE)
trainingmodelranfor 

#Get class predictions for training dataset
classtrainranfor <-  predict(trainingmodelranfor, type="raw")
head(classtrainranfor)
#Get class predictions for test dataset
classtestranfor <-  predict(trainingmodelranfor, newdata = df_test, type="raw")
head(classtestranfor)
#Derive predicted probabilites for test dataset
probsranfor=predict(trainingmodelranfor, newdata=df_test, type="prob")
head(probsranfor)
#Compute ROC 
rocranfor <- roc(df_test$READMISSION_INDICATOR,probsranfor[,"Yes"])
rocranfor

#The ROC curve
plot(rocranfor,col=c(3))
##Compute area under curve (Closer to 1 is preferred)
auc(rocranfor)
#confusionmatrix
confusionmatrixranfor<-confusion(predict(trainingmodelranfor, df_train), df_train$READMISSION_INDICATOR)
confusionmatrixranfor
# Overall Misclassification Rate
errorrateranfor<-(1-sum(diag(confusionmatrixranfor))/sum(confusionmatrixranfor))
errorrateranfor

#Sensitivity and Specificity
# Sensitivity - aka true positive rate, the recall, or probability of detection
sensitivityranfor<-sensitivity(confusionmatrixranfor)
sensitivityranfor
## Specificity - aka true negative rate
specificityranfor<-specificity(confusionmatrixranfor)
specificityranfor
#prediction
predranfor<-data.frame(classtestranfor)
predranfor <- cbind(df_test, predranfor)
#Write output of prediction to csv 
write.csv(predranfor, file = "Prediction_ranfor.csv")

#4. #CForest for Conditional Inference Tree using method="cforest"
#Assumptions
#a.No formal distributional assumptions, they are non-parametric and can thus handle skewed and multi-modal data.
#NOTE:- This particular algorithm may take a long time to run, may need to recalibrate using smaller samples
trainingmodelconfor <- train(as.factor(READMISSION_INDICATOR) ~ .,   data=df_train, method="cforest", trControl=cv_control)  
trainingmodelconfor

#Get class predictions for test dataset
classtestconfor <-  predict(trainingmodelconfor, newdata = df_test, type="raw")
head(classtestconfor)
#Derive predicted probabilites for test dataset
probsconfor=predict(trainingmodelconfor, newdata=df_test,  type="prob")
head(probsconfor)
#Compute ROC 
rocconfor <- roc(df_test$READMISSION_INDICATOR,probsconfor[,"Yes"])
rocconfor

#The ROC curve
plot(rocconfor,col=c(4))
##Compute area under curve (Closer to 1 is preferred)
auc(rocconfor)
#confusionmatrix
confusionmatrixconfor<-confusion(predict(trainingmodelconfor, df_train), df_train$READMISSION_INDICATOR)
confusionmatrixconfor
# Overall Misclassification Rate
errorrateconfor<-(1-sum(diag(confusionmatrixconfor))/sum(confusionmatrixconfor))
errorrateconfor
#Sensitivity and Specificity
# Sensitivity - aka true positive rate, the recall, or probability of detection
sensitivityconfor<-sensitivity(confusionmatrixconfor)
sensitivityconfor
## Specificity - aka true negative rate
specificityconfor<-specificity(confusionmatrixconfor)
specificityconfor
#prediction
predconfor<-data.frame(classtestconfor)
predconfor <- cbind(df_test, predconfor)
#Write output of prediction to csv 
write.csv(predconfor, file = "Prediction_confor.csv")

#5. #Random Forest with Boosting method="gbm"
#Assumptions
#a.No formal distributional assumptions, random forests are non-parametric and can thus handle skewed and multi-modal data.
modelLookup("gbm")
modelLookup("ada")
trainingmodelgbm <- train(as.factor(READMISSION_INDICATOR) ~ ., data=df_train, method="gbm",verbose=F,trControl=cv_control)
trainingmodelgbm 

#Get class predictions for training dataset
classtraingbm <-  predict(trainingmodelgbm, type="raw")
head(classtraingbm)

#Get class predictions for test datasey
classtestgbm <-  predict(trainingmodelgbm, newdata = df_test, type="raw")
head(classtestgbm)
#Get predicted probabilites for test dataset
probsgbm=predict(trainingmodelgbm, newdata=df_test, type="prob")
head(probsgbm)
#Compute ROC
rocgbm <- roc(df_test$READMISSION_INDICATOR, probsgbm[,"Yes"])
rocgbm 
#The ROC curve
plot(rocgbm, col=c(5))
##Compute area under curve (Closer to 1 is preferred)
auc(rocgbm)
#confusionmatrix
confusionmatrixgbm<-confusion(predict(trainingmodelgbm, df_train), df_train$READMISSION_INDICATOR)
confusionmatrixgbm
# Overall Misclassification Rate
errorrategbm<-(1-sum(diag(confusionmatrixgbm))/sum(confusionmatrixgbm))
errorrategbm
#Sensitivity and Specificity
# Sensitivity - aka true positive rate, the recall, or probability of detection
sensitivitygbm<-sensitivity(confusionmatrixgbm)
sensitivitygbm
## Specificity - aka true negative rate
specificitygbm<-specificity(confusionmatrixgbm)
specificitygbm
#prediction
predgbm<-data.frame(classtestgbm)
predgbm <- cbind(df_test, predgbm)
#Write output of prediction to csv 
write.csv(predgbm, file = "Prediction_gbm.csv")

#Final Model
##########################
###Random Forests Model### 
##########################  
vip_df<-df

#random sample half the rows 
newsample = sample(dim(vip_df)[1], dim(vip_df)[1]/2) # half of sample
#create training and test data sets
df_train = vip_df[newsample, ]
df_test = vip_df[-newsample, ]

#Random Forests
#Random Forest for Classification Trees using method="rf"
#Assumptions
#a.No formal distributional assumptions, random forests are non-parametric and can thus handle skewed and multi-modal data.
#I..Forward Propagation...
trainingmodelranfor <- train(as.factor(READMISSION_INDICATOR) ~ ., data=df_train,method="rf",trControl=cv_control, importance=TRUE)
trainingmodelranfor 
#Get class predictions for training dataset
classtrainranfor <-  predict(trainingmodelranfor, type="raw")
head(classtrainranfor)

#Get class predictions for test dataset
classtestranfor <-  predict(trainingmodelranfor, newdata = df_test, type="raw")
head(classtestranfor)
#Derive predicted probabilites for test dataset
probsranfor=predict(trainingmodelranfor, newdata=df_test, type="prob")
head(probsranfor)
#Compute ROC 
rocranfor <- roc(df_test$READMISSION_INDICATOR,probsranfor[,"Yes"])
rocranfor
#The ROC curve
plot(rocranfor,col=c(3))
##Compute area under curve (Closer to 1 is preferred)
auc(rocranfor)
#confusionmatrix
confusionmatrixranfor<-confusion(predict(trainingmodelranfor, df_train), df_train$READMISSION_INDICATOR)
confusionmatrixranfor
#Overall Misclassification Rate
errorrateranfor<-(1-sum(diag(confusionmatrixranfor))/sum(confusionmatrixranfor))
errorrateranfor
#Sensitivity and Specificity
# Sensitivity - aka true positive rate, the recall, or probability of detection
sensitivityranfor<-sensitivity(confusionmatrixranfor)
sensitivityranfor
## Specificity - aka true negative rate
specificityranfor<-specificity(confusionmatrixranfor)
specificityranfor
#prediction
predranfor<-data.frame(classtestranfor)
predranfor <- cbind(df_test, predranfor)
probsranfor_likelihood <-data.frame(probsranfor)
predranfor_train <- cbind(predranfor,probsranfor_likelihood)
#Write output of prediction to csv 
write.csv(predranfor_train, file = "Prediction_ranfor2a.csv")

#II..Backward Propagation...
testmodelranfor <- train(as.factor(READMISSION_INDICATOR) ~ ., data=df_test,method="rf",trControl=cv_control, importance=TRUE)
testmodelranfor 
#Get class predictions for test dataset
classtestranfor <-  predict(testmodelranfor, type="raw")
head(classtestranfor)
#Get class predictions for training dataset
classtrainranfor <-  predict(testmodelranfor, newdata = df_train, type="raw")
head(classtrainranfor)
#Derive predicted probabilites for training dataset
probsranfor=predict(testmodelranfor, newdata=df_train, type="prob")
head(probsranfor)
#Compute ROC 
rocranfor <- roc(df_train$READMISSION_INDICATOR,probsranfor[,"Yes"])
rocranfor
#The ROC curve
plot(rocranfor,col=c(3))
##Compute area under curve (Closer to 1 is preferred)
auc(rocranfor)
#confusionmatrix
confusionmatrixranfor<-confusion(predict(testmodelranfor, df_test), df_test$READMISSION_INDICATOR)
confusionmatrixranfor
# Overall Misclassification Rate
errorrateranfor<-(1-sum(diag(confusionmatrixranfor))/sum(confusionmatrixranfor))
errorrateranfor
#Sensitivity and Specificity
# Sensitivity - aka true positive rate, the recall, or probability of detection
sensitivityranfor<-sensitivity(confusionmatrixranfor)
sensitivityranfor
## Specificity - aka true negative rate
specificityranfor<-specificity(confusionmatrixranfor)
specificityranfor
#prediction
predranfor<-data.frame(classtrainranfor)
predranfor <- cbind(df_train, predranfor)
probsranfor_likelihood <-data.frame(probsranfor)
predranfor_test <- cbind(predranfor,probsranfor_likelihood)
#Write output of prediction to csv 
write.csv(predranfor_test, file = "Prediction_ranfor2b.csv")

#Rename column where names is...
names(predranfor_test)[names(predranfor_test) == "classtrainranfor"] <- "PREDICTION"
names(predranfor_train)[names(predranfor_train) == "classtestranfor"] <- "PREDICTION"

#Row bind the two dataframes
predranfor_all<-rbind(predranfor_test,predranfor_train)

#Recall other datapoints from original dataset
pre_df<-readm_data[,c(1:6)]

#Column bind the new two dataframes
Final_Result_readmission<-cbind(pre_df,predranfor_all)

#Write output of prediction to csv 
write.csv(Final_Result_readmission, file = "Final_Result_readmission.csv")

#Marketing Outreach to - Current Patients with No readmission but high likelihood of being readmitted in the future
Outreach_Population<-sqldf("select *
from Final_Result_readmission where READMISSION_INDICATOR='No' and PREDICTION='Yes'")

#Write output of prediction to csv 
write.csv(Outreach_Population, file = "Outreach_Population.csv")

#Recall Patient Outreach Data
#Relevant_var_Popn<-sqldf("select '1' as Key, FIRST,EMAILADDRESS,MAILINGADDRESS,PHONE_NUMBER from Outreach_Population")
Relevant_var_Popn<-sqldf("select PATIENT_FHIR_ID,PHONE,Yes*100 as PROPENSITYSCORE from Outreach_Population")
Relevant_var_Popn$DATESCORE <- format(Sys.time(), "%Y-%m-%d")

#Marketing & Outreach Data
Market_Outreach_Call<-Relevant_var_Popn

#Write output of prediction to csv 
write.csv(Market_Outreach_Call, file = "Prep_for_Market_Outreach_Readmission.csv")
#Creating the data for JSON file
jsonData <- toJSON(Market_Outreach_Call)
write(jsonData,"Prep_for_Market_Outreach_Readmission.json")

#Upload Model results data into container in Azure
cont_upload <- blob_container(blob_endp, "readmissionmodel")
upload_blob(cont_upload, src="C:\\Users\\olajideajayi\\OneDrive - Microsoft\\Documents\\Prep_for_Market_Outreach_Readmission.json")
upload_blob(cont_upload, src="C:\\Users\\olajideajayi\\OneDrive - Microsoft\\Documents\\Prep_for_Market_Outreach_Readmission.csv")

#Remove Azure Credentials from environment after use 
rm(Azure_SubID) 
rm(Azure_Storage_Key)
rm(Azure_tenantID)
rm(Azure_ResourceGrp)
