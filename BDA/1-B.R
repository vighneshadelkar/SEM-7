# (B) Write a R program to create a Data Frame which contain details of 5 employees and display
# summary of the data using R.

data<- data.frame(
Category= c("A","B","D","E","F"),
Age= c(10,15,20,30,5),
Values= c(1,2,3,0,10))

summary(data)
