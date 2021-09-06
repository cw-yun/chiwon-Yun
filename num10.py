prime_number = list(range(0,2000001))
sum = 0

for i in range(2000001):
    prime_number[i] = i

for i in range(2, 2000001):
    if prime_number[i] == 0 :
        continue
    for j in range(i+i,2000001,i):
        prime_number[j] = 0

for i in range(2,2000001):
    if prime_number[i] != 0 :
        sum += prime_number[i]

print(sum)