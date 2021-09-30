library(readxl)

accdata = read_excel("OT_fMRI_all data.csv.xls",sheet="OT_fMRI_all data")
psydata = read_excel("OT_questionnaire data.xlsx")
trtdata = read_excel("Participants_list_OT_fMRI_new.xlsx", sheet="Sheet1")
pcadata = read.csv("pca.csv")

accdata = as.data.frame(accdata)
psydata = as.data.frame(psydata)
trtdata = as.data.frame(trrtdata)
pcadata = as.data.frame(pcadata)

psydata$Index = psydata$Index%%1000

sumc = array(data=0,dim=psydata$Index[length(psydata$Index)]-psydata$Index[1]+1)
ncs = array(data=0,dim=psydata$Index[length(psydata$Index)]-psydata$Index[1]+1)
suma = array(data=0,dim=psydata$Index[length(psydata$Index)]-psydata$Index[1]+1)
nas = array(data=0,dim=psydata$Index[length(psydata$Index)]-psydata$Index[1]+1)

for (index in psydata$Index){
  for (j in 1:dim(accdata)[1]){
    if (accdata$sub[j]==index & (accdata$cond_id[j]==1|accdata$cond_id[j]==2)){
      sumc[index-104] = sumc[index-104]+accdata$STIM.ACC[j]
      ncs[index-104] = ncs[index-104]+1
    } else if (accdata$sub[j]==index & (accdata$cond_id[j]==3|accdata$cond_id[j]==4)){
      suma[index-104] = suma[index-104]+accdata$STIM.ACC[j]
      nas[index-104] = nas[index-104]+1
    }
  }
}

c_acc = sumc/ncs
a_acc = suma/nas

treatment = array(data=NA, dim=length(psydata$Index))
child_acc = array(data=NA, dim=length(psydata$Index))
adult_acc = array(data=NA, dim=length(psydata$Index))
avg_pc1 = array(data=NA, dim=length(psydata$Index))
avg_pc2 = array(data=NA, dim=length(psydata$Index))

for (i in 1:length(psydata$Index)) {
  index = psydata$Index[i]
  t = trtdata$Drug_name[trtdata$ID==index]
  if (t=="OT"){
    treatment[i] = 1
  }else {
    treatment[i] = 0
  }
  child_acc[i] = c_acc[index-104]
  adult_acc[i] = a_acc[index-104]
  avg_pc1[i] = mean(pcadata$X0[((i-1)*8+1):(i*8)])
  avg_pc2[i] = mean(pcadata$X1[((i-1)*8+1):(i*8)])
  
}



psydata$Treatment = treatment
psydata$Child_acc = child_acc
psydata$Adult_acc = adult_acc
psydata$PC1 = avg_pc1
psydata$PC2 = avg_pc2

write.csv(psydata, "data.csv", row.names = FALSE)
