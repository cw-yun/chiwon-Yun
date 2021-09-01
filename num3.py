num = 600851475143
count = 0

for i in range(num):
    if(i!=0):
       if(num % i ==0):
           for j in range(i):
               if(j!=0):
                  if(i%j==0):
                      count = count +1
           if(count==1):
               print(i)
           count = 0