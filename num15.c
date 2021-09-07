/*
Starting in the top left corner of a 2×2 grid, and only being able to move to the right and down, there are exactly 6 routes to the bottom right corner.

How many such routes are there through a 20×20 grid?
*/

#include <stdio.h>

int main()
{
    double route[21][21]={
    {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    };
    
    for(int i=0; i<20; i++)               // i means row
    {
        for(int j=0; j<20; j++)           // j means column
        {
            route[i+1][j+1] = route[i][j+1] + route[i+1][j];
        }
    }
    
    printf("%f", route[20][20]);
    return 0;
}

// answer : 137846528820
