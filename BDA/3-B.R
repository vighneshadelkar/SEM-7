# (B) Write the script in R to sort the values contained in the following vector in ascending order
# and descending order: (23, 45, 10, 34, 89, 20, 67, 99). Demonstrate the output using graph.

data <- c(23, 45, 10, 34, 89, 0, 67, 99)

# Ascending order
asc <- sort(data)
print(asc)

# Descending order
desc <- sort(data, decreasing = TRUE)
print(desc)

plot(asc, type = 'l')
plot(desc, type = 'l')

