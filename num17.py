dic={0:0,1:3,2:3,3:5,4:4,5:4,6:3,7:5,8:5,9:4,10:3,
     11:6,12:6,13:8,14:8,15:7,16:7,17:9,18:8,19:8,20:6,
     30:6,40:5,50:5,60:5,70:7,80:6,90:6,100:7,1000:11}
sum = 0

for i in range(1,1001):
    if len(str(i)) == 1:             # 1 digit
       sum += dic.get(i)

    elif len(str(i)) == 2:           # 10 digit
        if dic.get(i,0) == 0:        # dic.get(i,0) means if i can't be found in dic, get 0(ex:21,22,...etc)
            sum += dic.get(int(str(i)[0])*10)+dic.get(int(str(i)[1]))
        else :                        # (ex:10,11,12...etc)
            sum += dic.get(i)

    elif len(str(i)) == 3:           # 100 digit
        if i % 100 == 0:             # 100,200,... have no and
            num = dic.get(int(str(i)[0])) + dic.get(100)      # dic.get(100) is 7 (7 means hundred)
        else:                        # except 100,200,...
            num = dic.get(int(str(i)[0])) + dic.get(100) + 3  # plus and

        if dic.get(int(str(i)[1:]),0) == 0:                   # if 10 digit number is not in the dic
            sum += dic.get(int(str(i)[1])*10)+dic.get(int(str(i)[2])) + num
        else:                                                 # if 10 digit number is in the dic
            sum += dic.get(int(str(i)[1:])) + num

    else:                             # 1000
        sum += dic.get(i)

print(sum)