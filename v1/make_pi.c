#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define divisions 20000

int main ()
{
    double a[(divisions + 1)], b[divisions], c[divisions], trap = 0;
    double d[divisions], e[divisions], mid = 0, simp = 0, inc;
    int i, k;

    /*This program calculates pi using a Simpson's Rule estimation of the
      integral of arctangent from 0 to 1. In order to get more or less 
      accurate values, feel free to change the number of divisions in the
      #define up at the top. As given, this estimation of pi is guaranteed
      accurate up to 8 decimal places.*/

    /*First, the trapezoid rule is used to estimate pi.*/

    k=0;
    for (i=0;i<=divisions;i++)
    {
        a[i] = (double)k / (double)divisions;
        k++;
    }

    for (i=0;i<divisions;i++)
    {
        b[i] = pow (((a[i] + a[i+1]) / 2.0),2.0);
    }

    for (i=0;i<divisions;i++)
    {
        c[i] = (1.0 / (1.0 + b[i]))  * (1.0 / divisions);
    }

    for (i=0;i<divisions;i++)
    {
        trap = trap + c[i];
    }

    /*Next, the midpoint rule is also used to estimate pi.*/

    inc = ((1.0 / divisions) / 2.0);
    for (i=0;i<divisions;i++)
    {
        d[i] = inc;
        inc = inc + (1.0 / divisions);
    }

    for (i=0;i<divisions;i++)
    {
        e[i] = (1.0 / (1.0 + (pow (d[i],2.0)))) * (1.0 / divisions);
    }

    for (i=0;i<divisions;i++)
    {
        mid = mid + e[i];
    }

    /*Finally, Simpson's Rule is applied.*/

    simp = (((2.0 * mid) + trap) / 3.0) * 4.0;

    printf("\n\nThe value of pi is %.8lf.\n\n", simp);

return 0;
}
