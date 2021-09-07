count = 0
chain = 0

for i in range(1000001):
    while i != 1:
        if i%2 == 0 :
            i = i / 2
        else:
            i = 3*i + 1
        count += 1

    if chain < count:
        chain = count
        max = i
    count = 0

print(max)