//The prime factors of 13195 are 5, 7, 13 and 29.
//What is the largest prime factor of the number 600851475143 ?

#include <stdio.h>
#include <time.h>

int main()
{
    
    int count=0;
    long long num=600851475143;
    
    clock_t start = clock();
    
    for(long long i=2; i<600851475143; i++)
    {
       while(num % i == 0)
       {
           num = num / i;
           printf("%llu\n", i); // print all prime fator
       }
    }
    printf("result : %llu\n", num); // check perfectly prime factorization, print 1 means perfectly prime factorization
    clock_t end = clock(); // 60.51 minute
    printf("Time: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
    
    return 0;
}
    
// answer : 6857
// original time : unmeasureable
// after correction : 60.51 minute
