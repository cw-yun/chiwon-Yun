//A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.
//Find the largest palindrome made from the product of two 3-digit numbers.

#include <stdio.h>

int main()
{
    int a,b,c=0;
    for(int i=100; i<1000; i++)
    {
       for(int j=100; j<1000; j++)
       {
           a = i * j;
           while(a)
           {
               b = (b*10) + (a%10);
               a = a / 10;
           }
           
           a = i * j; // reset
           if((a==b) && (a>c))
           {
               printf("%d * %d = %d\n", i,j,a);
               c = a;              
           }
           
           b = 0; // reset
       }
    }
    
    return 0;
}

//answer : 906609
