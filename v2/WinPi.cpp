/*This program calculates pi using a Simpson's Rule estimation of the
integral of arctangent from 0 to 1. When inputting the number of
iterations to perform, more iterations = more precision. The number of
iterations is given as a command line argument. If no argument is
provided, a default value of 20,000 is used. At 20,000 iterations, the
value of pi is guaranteed to be accurate up to 8 decimal places. This
new multi-threaded implementation uses the Win32 API for threading. 
To compile, you'll need a C compiler, like the one that comes in 
Visual Studio; Windows doesn't compile via command line. The number of
threads to use is given as a command line argument. If no argument is
provided, a default value of 8 is used. For best performance, the
number of threads should not exceed the number of cores you have
available, and it should also divide evenly into the number of iterations
you input. To run this, just give it the iterations and threads arguments:
WinPi.exe 20000 8

In Visual Studio, these are usually found in a separate stdafx.h: 
#include <stdio.h>
#include <tchar.h>
#include <stdlib.h>
#include <time.h>
#include <process.h>
#include <Windows.h>
You may also have to mess with the project settings a little bit, 
otherwise you'll see some weird errors.  Specifically, turn off any 
Unicode Character Sets, because it'll mess up the command-line argument. 
*/

#include "stdafx.h"

// Global results arrays
long double * globaltrap;
long double * globalmid;

// Thread function pointer
unsigned __stdcall calculate(void *);

// Object to hold iteration values
typedef struct {
    long int threadid;
    long int lowlimit;
    long int highlimit;
    long int totaliterations;
} limits;

int _tmain(int argc, char* argv[])
{
    // Obtain command line arguments
    long int iterations = 20000L;
    if (argc > 2) {
        iterations = atol(argv[1]);
        if (iterations < 1L) {
            iterations = 20000L;
        }
    }
    int num_threads = 8;
    if (argc > 2) {
        num_threads = atoi(argv[2]);
        if (num_threads < 1) {
            num_threads = 8;
        }
    }

    // Initialize global storage
    int i;
    globaltrap = (long double *)calloc(num_threads, sizeof(long double));
    globalmid = (long double *)calloc(num_threads, sizeof(long double));
    uintptr_t * t_handles = (uintptr_t *)calloc(num_threads, sizeof(uintptr_t));
    limits ** funct_args = (limits **)calloc(num_threads, sizeof(limits *));
    if (globaltrap == 0 || globalmid == 0 || t_handles == 0 || funct_args == 0) {
        printf("Error allocating memory. Now exiting.\n");
        return -1;
    }
    for (i = 0; i < num_threads; i++) {
        funct_args[i] = (limits *)calloc(1, sizeof(limits));
        if (funct_args[i] == 0) {
            printf("Error allocating memory. Now exiting.\n");
            return -1;
        }
    }

    // Split off worker threads. When dividing the work, if the number of 
    // threads does not evenly divide into the desired number of iterations,
    // give any extra iterations to the final thread. This gives the final
    // thread at most (num_threads - 1) extra iterations. 
    long int clock_start = (long int)clock();
    for (i = 0; i < num_threads; i++) {
        funct_args[i]->threadid = (long int)i;
        funct_args[i]->lowlimit = (long int)i * (iterations / (long int)num_threads);
        funct_args[i]->highlimit = (((i + 1) == num_threads) ? iterations :
            ((long int)(i + 1) * (iterations / (long int)num_threads)));
        funct_args[i]->totaliterations = iterations;
        t_handles[i] = _beginthreadex(NULL, 0, calculate, funct_args[i], CREATE_SUSPENDED, NULL);
        if (t_handles[i] < 0) {
            printf("Error creating thread. Now terminating.\n");
            return -2;
        }
        ResumeThread((HANDLE)t_handles[i]);
    }

    // Wait for all the threads to return and check them
    for (i = 0; i < num_threads; i++) {
        WaitForSingleObject((HANDLE)t_handles[i], INFINITE);
        CloseHandle((HANDLE)t_handles[i]);
    }

    // After worker threads end, clean up each of the partial sums
    long double mid = 0.0L;
    long double trap = 0.0L;
    for (i = 0; i < num_threads; i++) {
        trap = trap + globaltrap[i];
        mid = mid + globalmid[i];
    }
    long int clock_end = (long int)clock();

    // Free global storage
    free(globaltrap);
    free(globalmid);
    free(t_handles);
    for (i = 0; i < num_threads; i++) {
        free(funct_args[i]);
    }
    free(funct_args);

    // Finally, Simpson's Rule is applied
    long double simp = (((2.0L * mid) + trap) / 3.0L) * 4.0L;
    printf("The calculated value of pi is %.21Lf\n", simp);
    printf("The actual value of pi is     3.141592653589793238463\n");
    printf("The time taken to calculate this was %.2f seconds\n",
        ((float)(clock_end - clock_start)) / (float)CLOCKS_PER_SEC);
    return 0;
}

// Function executed by each thread to incrementally calculate the overall value
unsigned __stdcall calculate(void * args)
{
    // Initialize needed variables
    limits * funct_args = (limits *)args;
    long double trap = 0.0L, mid = 0.0L;
    long double inverseiterations = 1.0L / (long double)funct_args->totaliterations;
    long int i;
    long int k = funct_args->lowlimit;
    long double inc = (inverseiterations / 2.0L) + inverseiterations * (long double)k;

    for (i = funct_args->lowlimit; i < funct_args->highlimit; i++) {
        // First, the trapezoid rule is used to estimate pi
        long double leftrect = (long double)k * inverseiterations;
        k++;
        long double rightrect = (long double)k * inverseiterations;
        long double trapsquared = ((leftrect + rightrect) / 2.0L) * ((leftrect + rightrect) / 2.0L);
        long double traptemp = (1.0L / (1.0L + trapsquared)) * inverseiterations;
        trap = trap + traptemp;

        // Next, the midpoint rule is also used to estimate pi
        long double inctemp = inc;
        inc = inc + inverseiterations;
        long double midtemp = (1.0L / (1.0L + (inctemp * inctemp))) * inverseiterations;
        mid = mid + midtemp;
    }

    // Save partial result and exit
    globaltrap[(funct_args->threadid)] = trap;
    globalmid[(funct_args->threadid)] = mid;
    return 0;
}
