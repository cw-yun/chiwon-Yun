day = 2
sum = 0
for year in range(1901,2001):
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        for month in range(1,13):
            if month == 4 or month == 6 or month == 9 or month == 11:
                day += 30
                if day % 7 == 0:
                    sum += 1
            elif month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
                day += 31
                if day % 7 == 0:
                    sum += 1
            elif month == 2:
                day += 29
                if day % 7 == 0:
                    sum += 1
    else:
        for month in range(1, 13):
            if month == 4 or month == 6 or month == 9 or month == 11:
                day += 30
                if day % 7 == 0:
                    sum += 1
            elif month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
                day += 31
                if day % 7 == 0:
                    sum += 1
            elif month == 2:
                day += 28
                if day % 7 == 0:
                    sum += 1
print(sum)