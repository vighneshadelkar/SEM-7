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

boxplot(subset)
