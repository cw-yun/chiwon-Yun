/*
Let d(n) be defined as the sum of proper divisors of n (numbers less than n which divide evenly into n).
If d(a) = b and d(b) = a, where a ≠ b, then a and b are an amicable pair and each of a and b are called amicable numbers.

For example, the proper divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110; therefore d(220) = 284. The proper divisors of 284 are 1, 2, 4, 71 and 142; so d(284) = 220.

Evaluate the sum of all the amicable numbers under 10000.
*/

#include <stdio.h>

int main()
{
    int sum = 0;
    int divisor_sum[10000] = {0};
    
    for(int i=1; i < 10000; i++)
    {
        for(int j=1; j < i; j++)
        {
            if(i % j == 0)
            {
                divisor_sum[i] += j;
            }
            
        }
    }
    
    for(int i=1; i < 10000; i++)
    {
        for(int j=1; j < 10000; j++)
        {
            if((divisor_sum[j] == i) && (divisor_sum[i] == j) && (i != j))
            {
               sum += i;
            }
        }
    }
    
    printf("%d\n", sum);
    return 0;
}

// answer : 31626
