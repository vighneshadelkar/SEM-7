data <- c(23, 45, 10, 34, 89, 0, 67, 99)

# Ascending order
asc <- sort(data)
print(asc)

# Descending order
desc <- sort(data, decreasing = TRUE)
print(desc)

plot(asc, type = 'l')
plot(desc, type = 'l')

