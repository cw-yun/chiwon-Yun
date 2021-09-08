/*
If the numbers 1 to 5 are written out in words: one, two, three, four, five, then there are 3 + 3 + 5 + 4 + 4 = 19 letters used in total.

If all the numbers from 1 to 1000 (one thousand) inclusive were written out in words, how many letters would be used?


NOTE: Do not count spaces or hyphens. For example, 342 (three hundred and forty-two) contains 23 letters and 115 (one hundred and fifteen) contains 20 letters. The use of "and" when writing out numbers is in compliance with British usage.
*/

#include <stdio.h>

int main()
{
    int count[1001] = {0};
    count[1] = 3; // one
    count[2] = 3; // two
    count[3] = 5; // three
    count[4] = 4; // four
    count[5] = 4; // five
    count[6] = 3; // six
    count[7] = 5; // seven
    count[8] = 5; // eight
    count[9] = 4; // nine
    count[10] = 3; // ten
    count[11] = 6; // eleven
    count[12] = 6; // twelve
    count[13] = 8; // thirteen
    count[14] = 8; // fourteen
    count[15] = 7; // fifteen
    count[16] = 7; // sixteen
    count[17] = 9; // seventeen
    count[18] = 8; // eighteen
    count[19] = 8; // nineteen
    count[20] = 6; // twenty
    count[30] = 6; // thirty
    count[40] = 5; // forty
    count[50] = 5; // fifty
    count[60] = 5; // sixty
    count[70] = 7; // seventy
    count[80] = 6; // eighty
    count[90] = 6; // ninety
    count[100] = 10; // one hundred
    count[200] = 10; // two hundred
    count[300] = 12; // three hundred
    count[400] = 11; // four hundred
    count[500] = 11; // five hundred
    count[600] = 10; // six hundred
    count[700] = 12; // seven hundred
    count[800] = 12; // eight hundred
    count[900] = 11; // nine hundred
    count[1000] = 11; // one thousand
    
    int sum = 0;
        
    for(int i = 1; i <= 1000; i++)
    {
        if(count[i] == 0)                  // number is not in the upper count[]
        {
            if(i < 100)
            {
                count[i] = count[(i/10)*10] + count[i%10];
            }
            
            else
            {
                if(i % 100 == 0)
                {
                    count[i] = count[i];
                }
                else
                {
                    if(((i>110) && (i<120)) || ((i>210) && (i<220)) || ((i>310) && (i<320)) || ((i>410) && (i<420)) || ((i>510) && (i<520)) || ((i>610) && (i<620)) || ((i>710) && (i<720)) || ((i>810) && (i<820)) || ((i>910) && (i<920)))
                    {
                        count[i] = count[(i/100)*100] + count[(i%100)] + 3;                              // 3 means and
                    }
                    else
                    {
                        count[i] = count[(i/100)*100] + count[((i/10)%10)*10] + count[(i%100)%10] + 3;   // 3 means and
                    }
                }
            }
        }
    }
    
    for(int i = 1; i <= 1000; i++)
    {
        sum += count[i];
    }
    printf("%d\n", sum);
    return 0;
}

// answer : 21124
