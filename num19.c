/*
You are given the following information, but you may prefer to do some research for yourself.

1 Jan 1900 was a Monday.
Thirty days has September,
April, June and November.
All the rest have thirty-one,
Saving February alone,
Which has twenty-eight, rain or shine.
And on leap years, twenty-nine.
A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.
How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?
*/

#include <stdio.h>

int main()
{
    int year;
    int month;
    int day = 2;
    int sum = 0;
    
    for(year=1901; year <= 2000; year++)
    {
       if((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0))          // leap year
       {
           for(month = 1; month <= 12; month++)
           {
               if((month == 4) || (month == 6) || (month == 9) || (month == 11))
               {
                   day += 30;
                   if(day % 7 == 0)
                   {
                       sum++;
                   }
               }
               
               else if((month == 1) || (month == 3) || (month == 5) || (month == 7) || (month == 8) || (month == 10) || (month == 12))
               {
                   day += 31;
                   if(day % 7 == 0)
                   {
                       sum++;
                   }
               }
               
               else if(month == 2)
               {
                   day += 29;
                   if(day % 7 == 0)
                   {
                       sum++;
                   }
               }
           }
       }
       
       else                                                                 // no leap year
       {
           for(month = 1; month <= 12; month++)
           {
               if((month == 4) || (month == 6) || (month == 9) || (month == 11))
               {
                   day += 30;
                   if(day % 7 == 0)
                   {
                       sum++;
                   }
               }
               
               else if((month == 1) || (month == 3) || (month == 5) || (month == 7) || (month == 8) || (month == 10) || (month == 12))
               {
                   day += 31;
                   if(day % 7 == 0)
                   {
                       sum++;
                   }
               }
               
               else if(month == 2)
               {
                   day += 28;
                   if(day % 7 == 0)
                   {
                       sum++;
                   }
               }
           }
       }
    }
    
    printf("%d\n", sum);
    return 0;
}

// answer : 171
