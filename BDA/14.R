# 1) create data frame for followinf 4 vectors
# 2) display structure and summary of above data
# 3) extract emp_name and salary from the anove data structure
# 4) extract the employee details whose salary is less than or equal to 60000

emp_id=c(1:5)
emp_name=c("a","b","c","d","e")
start_date=c(1,1,1,1,1)
salary=c(60000,45000,75000,84000,200000)

data=data.frame(emp_id,emp_name,start_date,salary)

summarized_data=summary(data)
print(summarized_data)

print(data$emp_id)
print(data$emp_name)

filtered_data=subset(data,salary<=60000)
print(filtered_data)
