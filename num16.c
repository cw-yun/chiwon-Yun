/*
215 = 32768 and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26.

What is the sum of the digits of the number 21000?
*/

#include <stdio.h>
#include <math.h>

int main()
{
    double a = pow(2.0, 1000.0);             //  pow : exponential function
    char s[500];
    int sum = 0;
    
    sprintf(s, "%f", a);                     // convert formation double to string
    for(int i=0; i<500; i++)
    {
        if(s[i] == '.')                      // remove below decimal point
        {
            break;
        }
        sum += s[i] - 48;
    }
    printf("%d", sum);
    
    return 0;
}

// answer : 1366
