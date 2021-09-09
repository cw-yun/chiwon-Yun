divisor_sum = [0]
sum = 0
all_sum = 0

for i in range(1,10000):
    sum = 0
    for j in range(1, i):
        if i % j == 0:
            sum += j
    divisor_sum.append(sum)

for i in range(10000):
    for j in range(10000):
        if divisor_sum[i] == j and divisor_sum[j] == i and i != j:
            all_sum += i

print(all_sum)