count=0
num=0
for i in range(2,2147483647):
    for j in range(2,i):
        if i%j==0:
            count += 1
    if count == 1:
        num += 1
    if num == 10001:
        print(i)