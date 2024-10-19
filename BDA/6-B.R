# i) Which R command will Mr. John use to enter these values demonstrate the output.
# ii) Now Mr. John wants to add the salaries of 5 new employees in the existing table,
# which command he will use to join datasets with new values in R. Demonstrate the
# output.
# (iii) Visialize the data using chart .

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
