//The sum of the squares of the first ten natural numbers is, 1^2 + 2^2 + ... + 10^2 = 385
//The square of the sum of the first ten natural numbers is, (1 + 2 + ... + 10)^2 = 3025
//Hence the difference between the sum of the squares of the first ten natural numbers and the square of the sum is .
//Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.

#include <stdio.h>

int main()
{
    double sum, sum_of_square, square_of_sum=0;
    for(int i=1; i<=100; i++)
    {
        sum_of_square += i*i;
        sum += i;
    }
    square_of_sum = sum * sum;
    printf("sum_of_square : %f\nsquare_of_sum : %f\ndiffrence : %f\n", sum_of_square, square_of_sum, (square_of_sum - sum_of_square));
    
    return 0;
}
