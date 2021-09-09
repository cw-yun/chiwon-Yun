/*
n! means n × (n − 1) × ... × 3 × 2 × 1

For example, 10! = 10 × 9 × ... × 3 × 2 × 1 = 3628800,
and the sum of the digits in the number 10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.

Find the sum of the digits in the number 100!
*/

#include <stdio.h>

int main()
{
    int value[500] = {0};
    value[0] = 1;
    int remain;
    int upper;
    int sum = 0;
    int x;
    
    for(int i=1; i <= 100; i++)
    {
        upper = 0;
        for(int j=0; j < 500; j++)
        {
            x = value[j] * i + upper;
            upper = 0;
            if(x > 9)
            {
                remain = x % 10;
                upper = x / 10;
            }
            else
            {
                remain = x;
            }
            value[j] = remain;
        }
    }
    
    for(int i=499; i >= 0; i--)
    {
        sum += value[i];
    }
    
    printf("%d\n",sum);
    return 0;
}

// answer : 648
