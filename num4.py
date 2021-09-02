b = 0
c = 0
for i in range(100,1000):
    for j in range(100,1000):
        a=i*j
        while a:
            b = (b*10) + (a%10)
            a = int(a / 10)             #convert to int type
        a=i*j
        if a==b and a>c:
            print(i, '*', j, '=', a)
            c=a
        b=0
