a=0
b=1
sum =0
for i in range(50):
    c = a+b
    #print(c)
    a=b
    b=c
    if(c>0 and c<4000000 and c%2==0):
        sum += c
print(sum)