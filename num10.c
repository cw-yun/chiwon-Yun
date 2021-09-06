//The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
//Find the sum of all the primes below two million.

#include <stdio.h>

int main()                            // Sieve of Eratosthenes
{
    long long sum=0;
    int prime_number[2000001];
    
    for(int i=0; i<=2000000; i++)
    {
        prime_number[i]=i;        
    }
    
    for(int i=2; i <=2000000; i++)
    {
        if(prime_number[i] == 0)
        {
            continue;
        }
        
        for(int j=i+i; j <= 2000000; j += i)
        {
            prime_number[j] = 0;
        }
     }
     
     for(int i=2; i <= 2000000; i++)
     {
         if(prime_number[i] != 0)
         {
             sum += prime_number[i];
         }
     }
    
    printf("%llu\n", sum);
    return 0;
}

// answer : 142913828922

