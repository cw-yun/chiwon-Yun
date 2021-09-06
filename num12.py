i = 0
num = 0
count = 0

while 1:
    i += 1
    num += i
    for j in range(1, num):
        if num%j == 0:
            count += 1

    if count >= 500:
        print(num)
        break
    else:
        count = 0
