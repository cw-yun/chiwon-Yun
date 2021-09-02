//By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.
//What is the 10001st prime number

#include <stdio.h>

int main()
{
    int count =0;
    int num=0;
    for(int i=2; i<=2147483647; i++)    // range of int type   // while로 수정해서 풀어보세요 ~ 
    {
        for(int j=2; j<=i; j++)
        {
            if(i%j==0)
            {
                count++;
            }
        }
        if(count==1)
        {
             num ++;
        }
        if(num==10001)
        {
            printf("%d\n", i);
            break;
        }
        count = 0;
    }
    
    return 0;
}

// answer : 104743
