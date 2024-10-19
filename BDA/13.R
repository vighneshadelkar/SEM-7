# i. Create a subset of course less than 3 by using [ ] brackets and demonstrate the
# output.
# ii. Create a subset where the course column is less than 3 or the class equals to 2
# by using subset () function and demonstrate the output.

data=data.frame(
    course=c(1,2,3,4,5,6),
    id=c(11,12,13,14,15,16),
    class=c(1,2,1,2,1,2),
    marks=c(56,75,48,69,84,53)
)

course_subset=data[data$course<3,] 

course_subset2=subset(data,course<3 | class==2)
print(course_subset)
print(course_subset2)
