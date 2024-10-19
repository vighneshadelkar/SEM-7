salaries <- data.frame(
  name = c("John", "Mary", "David", "Emily", "James"),
  salary = c(50000, 60000, 70000, 80000, 90000) 
)

new_employees <- data.frame(
  name = c("Alice", "Bob"),
  salary = c(10000, 20000)
)

updated_employees <- rbind(salaries, new_employees)
print(updated_employees)
