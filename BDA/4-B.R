bread<-c(12,3,4,11,9)
milk<-c(71,27,18,20,15)
cola<-c(10,1,33,6,12)
chocolate<-c(6,7,4,13,12)
detergent<-c(5,8,12,20,23)

data<-data.frame(bread,milk,cola,chocolate,detergent)

barplot(as.matrix(data),beside=TRUE,col=rainbow(5),main="product sales by day")
