sum = 0
sum_of_square = 0
for i in range(1,101):
    sum = sum + i
    sum_of_square = sum_of_square + i*i
square_of_sum = sum * sum
print("difference : ", square_of_sum-sum_of_square)