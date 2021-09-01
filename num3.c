//The prime factors of 13195 are 5, 7, 13 and 29.
//What is the largest prime factor of the number 600851475143 ?

#include <stdio.h>
#include <time.h>

int main()
{
    
    int count=0;
    long long num=600851475143;
    
    //clock_t start = clock();
    
    for(long long i=1; i<600851475143; i++)
    {
       if(num % i == 0)
       {
           for(long long j=1; j<=i; j++)
           {
               if(i%j==0)
               {
                   count++;
               }
           }
           if(count == 2)
               printf("%llu\n", i);
           count = 0;
       }
    }
    //clock_t end = clock();
    //printf("Time: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
    
    return 0;
}
    
// answer : 6857
