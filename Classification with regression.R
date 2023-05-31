#### LOADING THE DATASETS ####

gcopy <- read.csv('~/Speciale/gcopy.csv')
gu6 <- read.csv('~/Speciale/gu6.csv')
g6to10 <- read.csv('~/Speciale/g6to10.csv')
g10to20 <- read.csv('~/Speciale/g10to20.csv')
go20 <- read.csv('~/Speciale/go20.csv')

#MARKING VARIABLES IN DATASET
Image1 <- gcopy$Image1
Image2 <- gcopy$Image2
Image3 <- gcopy$Image3
Image4 <- gcopy$Image4
Image5 <- gcopy$Image5

#### TEST FOR UNIVARIATE NORMALITY ####

# Making a function to plot the QQ-plot and to print the rq-value (called r in the plot)
qq_function <- function(x, name_variable){
  qq_plot <- qqnorm(x, plot.it = F)
  my_cor <- cor(qq_plot$x, qq_plot$y) # step 4
  plot(qq_plot,
       main = paste0("Q-Q plot for ", name_variable),
       ylab = "Observed quantiles",
       xlab = "Theoretical quantiles") # plot the data
  legend('topleft', paste0("r = ", round(my_cor,4))) # adds the correlation value to the chart
}


# TESTING THE UNIVARIATE DISTRIBUTION FOR EACH OF THE IMAGES ###
par(mfrow=c(2,3))
qq_function(Image1, "Image 1") #r=0.9865
qq_function(Image2, "Image 2") #r=0.9841
qq_function(Image3, "Image 3") #r0.9822
qq_function(Image4, "Image 4") #r= 0.979
qq_function(Image5, "Image 5") #r=0.983


#Remember that n = 179. We find the critical value ((1), table 4.2 p. 181), at α=0,05. This is between 0,9913 and 0,9931 – we use the approximate value r = 0,9922 and evaluate the results from the QQ-plots
#NONE OF THE ABOVE ARE NORMALLY DISTRIBUTED



##### TEST FOR BIVARIATE NORMALITY #####

#Function to find critical value 
nn <- nrow(gcopy) #n
pp <- 2 #number of variables
N0 <- 1000
FindcrikChi <- function(n=nn, p=pp, alpha=alpha1, N=N0){ #This is Jing's Script FindcrikChi
  cricvec <- rep(0, N) #vector for the rQ result collection#
  for(i in 1:N){
    #iteration to estimate rQ#
    numvec <- rchisq(n, p) #generate a data set of size n, degree of freedom=p#
    d <- sort(numvec)
    q <- qchisq((1:n-0.5)/n, p)
    cricvec[i] <- cor(d,q)
  }
  scricvec <- sort(cricvec)
  cN <- ceiling(N* alpha) #to be on the safe side I use ceiling instead of floor(), take the 'worst' alpha*N cor as rQ, everything lower than that is deemed as rejection#
  cricvalue <- scricvec[cN]
  result <- list(cN, cricvalue, scricvec)
  return(result)
}
resultsbi <- FindcrikChi(nn, pp, .05, 1000) #choosing 1000 iterations as the critical value does not change much after that. 2 coloums because it is a bivariate
resultsbi

#Critical value is 0.9805

#Make function to plot
bivar_norm <- function(x1, x2, alpha, name, remove_outlier = FALSE) {
df <- data.frame(x1,x2) # create dataframe
n <- nrow(df) # observations
p <- ncol(df) # number of variables
D2 <- mahalanobis(df,
               center = colMeans(df),
               cov = cov(df)) # generalized squared distance
if(remove_outlier == TRUE){
D2 <- D2[-which.max(D2)]
}
chi_plot <- qqplot(qchisq(ppoints(n, a = .5), df = p), D2,
               plot.it = F) # chi square plot values.
# ppoints: j-1/2/n = 1:length(x)-1/2/length(x)
my_cor <- cor(chi_plot$x, chi_plot$y) # correlation value
critical_value <- qchisq(p = alpha,
                     df = p,
                     lower.tail = F) # calculate critical value
prop_within_contour <- round(length(D2[D2 <= critical_value]) / length(D2),4)
plot(chi_plot,
ylab = 'Mahalanobis distances',
xlab = 'Chi-square quantiles',
main = paste0(name)) # plot chi square plot
legend("topleft",
   paste0("r = ", round(my_cor,4), "\n",
         "% D2 <= cˆ2: ", prop_within_contour, "\n",
         "Expected if normal: ", 1-alpha),
   cex = 0.75,
   bty = "n") # add legend to plot
}



par(mfrow=c(2,3))
bivar_norm(Image1,Image2, .05, "Image1 and Image2", F) #0.9709 #Not normal
bivar_norm(Image1,Image3, .05, "Image1 and Image3", F) #0.9934 #Normal
bivar_norm(Image1,Image4, .05, "Image1 and Image4", F) #0.991 #Normal
bivar_norm(Image1,Image5, .05, "Image1 and Image5", F) #0.9573 #Not normal
bivar_norm(Image2,Image3, .05, "Image2 and Image3", F) #0.9535 #Not normal
bivar_norm(Image2,Image4, .05, "Image2 and Image4", F) #0.9633 #Not normal
bivar_norm(Image2,Image5, .05, "Image2 and Image5", F) #0.9656 #Not normal
bivar_norm(Image3,Image4, .05, "Image3 and Image4", F) #0.909 #Not normal
bivar_norm(Image3,Image5, .05, "Image3 and Image5", F) #0.9633 #Not normal
bivar_norm(Image4,Image5, .05, "Image4 and Image5", F) #0.9423 #Not normal




#### TESTING MULTIVARIATE DISTRIBUTION (NORMAL) ON ALL FIVE IMAGES ####
#FUNCTION TO FIND CRITICAL VALUE
nn <- nrow(gcopy) #n#
pp <- 5 #p#
N0 <- 1000

resultmulti <- FindcrikChi(nn, pp, .05, 1000) #choosing 1000 iterations as the critical value does not change much after that. 2 coloums because it is a bivariate
resultmulti 

# We find the critical value at α=0,05 to be 0.9848

#FUNCTION TO MAKE PLOT
chi_square_all <- function(x1,x2,x3,x4,x5,alpha,name, remove_outlier = FALSE){
  df <- data.frame(x1,x2,x3,x4,x5) # create dataframe
  n <- nrow(df) # observations
  p <- ncol(df) # number of variables
  D2 <- mahalanobis(df,
                    center = colMeans(df),
                    cov = cov(df)) # generalized squared distance
  if(remove_outlier == TRUE ){
    D2 <- D2[-which.max(D2)]
  }
  chi_plot <- qqplot(qchisq(ppoints(n, a = .5), df = p), D2,
                     plot.it = F) # chi square plot values
  my_cor <- cor(chi_plot$x, chi_plot$y) # correlation value
  critical_value <- qchisq(p = alpha,
                           df = p,
                           lower.tail = F) # calculate critical value
  prop_within_contour <- round(length(D2[D2 <= critical_value]) / length(D2),4)
  plot(chi_plot,
       ylab = 'Mahalanobis distances',
       xlab = 'Chi-square quantiles',
       main = paste0('chi square plot of D2 vs. chi_2ˆ2(',alpha,") for ",
                     name)) # plot chi square plot
  legend("topleft",
         paste0("r = ", round(my_cor,4), "\n",
                "% D2 <= cˆ2: ", prop_within_contour, "\n",
                "Expected if normal: ", 1-alpha),
         cex = 1,
         bty = "n") # add legend to plot
}
par(mfrow=c(1,1))


#THE WHOLE DATASET
chi_square_all(Image1, Image2, Image3, Image4, Image5, .05, "All variables", F)
##r=0.929 #NOT NORMAL DISTRIBUTION


#### TEST FOR EQUAL COVARIANCE FOR THE FOUR SUBDATASETS ####
#Test Homogeneous covariance matrices
#Taking the subdatasets and deleting some of the coloums to only keep the coloumns with images sizes and pathology polyp size
gu6_1<-gu6[,3:8]
g6to10_1<-g6to10[,3:8]
g10to20_1<-g10to20[,3:8]
go20_1<-go20[,3:8]

g <- 4 #four classes#
p <- 5 #five variables#

#covariance for the four classes
s1 <- cov(gu6_1)
s2 <- cov(g6to10_1)
s3 <- cov(g10to20_1)
s4 <- cov(go20_1)
#number of rows in each class and summed
n1 <- nrow(gu6_1) #38
n2 <- nrow(g6to10_1) #28
n3 <- nrow(g10to20_1) #78
n4 <- nrow(go20_1) #35
n <- n1+n2+n3+n4 #179

#we start by calculating Spooled (6-49 p 310)
w <- (n1-1)*s1+(n2-1)*s2+(n3-1)*s3+(n4-1)*s4 #Within matrix#
spooled <- w/(n-g)
# Compute M (6-50 p311)
M <- (n-g)*log(det(spooled))-(n1-1)*log(det(s1))-(n2-1)*log(det(s2))-(n3-1)*log(det(s3)-(n4-1)*log(det(s4)))
# Compute correction factor (6-51 p. 311)
u <- (1/(n1-1)+1/(n2-1)+1/(n3-1)+1/(n4-1)-1/(n-g))*(2*p^2+3*p-1)/(6*(p+1)*(g-1))
# Test statistic (6-52 p. 311)
C <- (1-u)*M
# critical value #v=p*(p+1)*(g-1)/2# (6-53 p. 311)
critvalue <- qchisq(.95,p*(p+1)*(g-1)/2)
# final decision 
decisionflag <- (C > critvalue) #True, therefore therefore we should NOT accept the H0, i.e. not homogeneous covariance matrix#


#### CLASSIFICATION WITH MULTINOMIAL REGRESSION ####
#MAKING A NEW DATASET WHERE THE CATEGORY IS IN A COLOUMN
gu6['Category'] = 'Under 6 mm'
g6to10['Category'] = 'Between 6 and 10 mm'
g10to20['Category'] = 'Between 10 and 20 mm'
go20['Category'] = 'Over 20 mm'

dataframe3 = rbind(gu6, g6to10, g10to20, go20)
print(dataframe3)

# fit multinominal logistic discrimination ##script modified from "admission2022.R"
library(nnet)
y <- dataframe3$Category
multinomModel <- multinom(Category~Image1+Image2+Image3+Image4+Image5,dataframe3) #https://www.rdocumentation.org/packages/nnet/versions/7.3-18/topics/multinom
summary(multinomModel)
prlogi <- predict(multinomModel, dataframe3)
prlogi
T5 <- table(y, prlogi)
T5
#testing using APER
aper <- (sum(T5)-sum(diag(T5)))/sum(T5) #We take the sum of all observations and subtract the diagonal (which are correctly classified) and the divide by the number of all observations (11-34 p. 598)
aper
prcv <- rep(0,n) #We can capture the results of our loop in a list.First we create a vector and then we fill in its values.
for (i in 1:n){
  dataframeholdout <- dataframe3[-i,] #taking a row out of the model #step 1 one page 599 of the holdout procedure
  multiholdout <- multinom(Category~Image1+Image2+Image3+Image4+Image5,dataframeholdout) #calculating this in the model (where it would be classified)
  xc <- dataframe3[i,3:10] #the observation row that is taken out
  prcv[i]<- predict(multiholdout,xc) #predicting the outcome of the observation that is taken out #step 2 of the holdout model p. 599
}
# compute the CV error rate
T7 <- table(y, prcv)
T7
e_aper <- (sum(T7)-sum(diag(T7)))/sum(T7) #We take the sum of all observations and subtract the diagonal (which are correctly classified) and the divide by the number of all observations (11-34 p. 598)
e_aper


n = sum(T5) # number of instances
nc = nrow(T5) # number of classes
diag = diag(T5) # number of correctly classified instances per class 
rowsums = apply(T5, 1, sum) # number of instances per class
colsums = apply(T5, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes

accuracy = sum(diag) / n 
print(accuracy) #0.6425

precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1) 
weightedf1 =f1[1]*(n3/n)+f1[2]*(n2/n)+f1[3]*(n4/n)+f1[4]*(n1/n) #0,609



macroPrecision = mean(precision)
macroRecall = mean(recall)
macroF1 = mean(f1)

data.frame(macroPrecision, macroRecall, macroF1)

n = sum(T7) # number of instances
nc = nrow(T7) # number of classes
diag = diag(T7) # number of correctly classified instances per class 
rowsums = apply(T7, 1, sum) # number of instances per class
colsums = apply(T7, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes

accuracy = sum(diag) / n 
print(accuracy) #0.6425

precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
f1
data.frame(precision, recall, f1) 
weightedf1 =f1[1]*(n3/n)+f1[2]*(n2/n)+f1[3]*(n4/n)+f1[4]*(n1/n) #0,55



macroPrecision = mean(precision)
macroRecall = mean(recall)
macroF1 = mean(f1)

data.frame(macroPrecision, macroRecall, macroF1)

