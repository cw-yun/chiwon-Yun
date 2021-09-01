//Each new term in the Fibonacci sequence is generated by adding the previous two terms. By starting with 1 and 2, the first 10 terms will be:
//1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
//By considering the terms in the Fibonacci sequence whose values do not exceed four million, find the sum of the even-valued terms.

#include <stdio.h>

int main()
{
   int a = 0;
   int b = 1;
   int c = 0;
   int sum = 0;
   for(int i=0; i<50; i++)
   {
      c=a+b;
      if(c<4000000 && c>0){
         //printf("%d\n", c);
         if(c % 2 == 0 )
         sum = sum + c;
         }
      a = b;
      b = c;
   }
   printf("%d\n",sum);
   
   return 0;
}

// answer : 4613732