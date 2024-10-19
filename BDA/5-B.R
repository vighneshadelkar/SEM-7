# (i) Create a subset of subject less than 4 by using subset () funcon and demonstrate the output.
# (ii) Create a subject where the subject column is less than 3 and the class equals to 2 by using []
# brackets and demonstrate the output using R

data<-data.frame(
    Subject=c(1,2,3,4,5,6),
    Class=c(1,2,1,2,1,2),
    Marks=c(56,75,48,69,84,53)
)

# print(data)
subset<-subset(data,Subject<4)
print(subset)

subset_2<-data[data$Subject<3 & data$Class==2,]
print(subset_2)


barplot(as.matrix(data),beside=TRUE,col=rainbow(5))
