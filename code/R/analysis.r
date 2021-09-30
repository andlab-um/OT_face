library(randomForest)
library(Boruta)
library(caret)
library(ggtext)

fmriroot = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/fmri_data/fMRI_haiyan_processed/"


data = read.csv("data.csv")
data = na.omit(data)
set.seed(99)

child.acc = data$Child_acc
adult.acc = data$Adult_acc
trmt = data$Treatment

n.sub = dim(data)[1]
n.var = dim(data)[2]
pp = preProcess(data[2:(n.var-3)], method = c("BoxCox"), na.remove=T)
ppx = as.data.frame(scale(predict(pp, data[2:(n.var)])))

ppdata = as.data.frame(ppx)
ppdata$Child_acc = data$Child_acc
ppdata$Adult_acc = data$Adult_acc
ppdata$Treatment = trmt


pc1.boruta <- Boruta(PC1~.-PC2, data=ppdata,
                       doTrace = 0,
                       pValue = 0.05,
                       holdHistory = TRUE,
                       maxRuns = 10000)
png('PC1.png',width=2000, height=1900, res=300)
par(family="serif")
p <- plot(pc1.boruta, las = 2, cex.axis = 0.6, xlab="") #, main="Variable contribution to Principle Component 1"
dev.off()

pc1.boruta.var <- getSelectedAttributes(pc1.boruta, withTentative = TRUE)
print(pc1.boruta.var) 

pc2.boruta <- Boruta(PC2~.-PC1, data=ppdata,
                     doTrace = 0,
                     pValue = 0.05,
                     holdHistory = TRUE,
                     maxRuns = 10000)
plot(pc2.boruta, las = 2, cex.axis = 0.6, main="Var Imp - PC2",xlab="")
pc2.boruta.var <- getSelectedAttributes(pc2.boruta, withTentative = TRUE)
print(pc2.boruta.var) 

pca_rt_acc = read.csv("pca_rt_acc.csv")

summary(lm(pca~child_rt, data=pca_rt_acc))
summary(lm(pca~adult_rt, data=pca_rt_acc))
summary(lm(pca~child_acc, data=pca_rt_acc))
summary(lm(pca~adult_acc, data=pca_rt_acc))

for (i in names(data)[2:31]) {
  t = describeBy(data[i], data$Treatment, mat = TRUE)
  p = t.test(as.formula(paste0(i, " ~ Treatment")), data=data)
  message(i, " & ", round(t$mean[1],2), " (", round(t$sd[1],2),") & ", round(t$mean[2],2), " (",round(t$sd[2],2),") & ", 
          round(p$statistic,4)," & ", round(p$p.value,4), " \\\\")
}

