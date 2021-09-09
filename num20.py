mul = 1
sum = 0
for i in range(1,101):
    mul *= i

for i in range(len(str(mul))):
    sum += int(str(mul)[i])
print(sum)