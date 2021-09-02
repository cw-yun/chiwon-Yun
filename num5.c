//2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.
//What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?

#include <stdio.h>

int main()
{
   for(int i=2; i<2147483647; i++) // range of int type // 최대값을 모를때는 while 조건을 고려해 보세요 
   {
      if((i%11==0) && (i%12==0) && (i%13==0) && (i%14==0) && (i%15==0) && (i%16==0) && (i%17==0) && (i%18==0) && (i%19==0) && (i%20==0)) // 좀더 간결한 방법을 고민해 보세요 
      {
         printf("%d\n", i);
      }
   }
   return 0;
}

// answer : 232792560
