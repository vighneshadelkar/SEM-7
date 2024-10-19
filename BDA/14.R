emp_id=c(1:5)
emp_name=c("a","b","c","d","e")
start_date=c(1,1,1,1,1)
salary=c(60000,45000,75000,84000,200000)

data=data.frame(emp_id,emp_name,start_date,salary)
print(data)

print(data$emp_id)
print(data$emp_name)

filtered_data=subset(data,salary<=60000)
print(filtered_data)
