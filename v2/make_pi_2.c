/*This program calculates pi using a Simpson's Rule estimation of the
integral of arctangent from 0 to 1. When inputting the number of 
iterations to perform, more iterations = more precision. The number of
iterations is given as a command line argument. If no argument is 
provided, a default value of 20,000 is used. At 20,000 iterations, the
value of pi is guaranteed to be accurate up to 8 decimal places. This
new multi-threaded implementation uses the POSIX pthread library to
perform the parallel computations. Since this library is only available
on Unix/Linux, see WinPi for the windows version. The number of
threads to use is given as a command line argument. If no argument is
provided, a default value of 8 is used. For best performance, the 
number of threads should not exceed the number of cores you have 
available, and it should also divide evenly into the number of iterations
you input. To compile this code, run the following at the command line: 
    cc -O1 -Wall -c make_pi_2.c
    cc -lpthread -lrt -o make_pi_2 make_pi_2.o
    rm make_pi_2.o
Then to run it, just give it the iterations and threads arguments:
    make_pi_2 20000 8
*/

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// Global results arrays
long double * globaltrap;
long double * globalmid;

// Thread function pointer
void * calculate (void *);

// Object to hold iteration values
typedef struct {
    long threadid;
    long lowlimit;
    long highlimit;
    long totaliterations;
} limits;

// Main function
int main (int argc, char * argv[])
{
    // Obtain command line arguments
    long iterations = 20000L;
    if (argc > 1) {
        iterations = atol (argv[1]);
        if (iterations < 1L) {
            iterations = 20000L;
        }
    }
    long num_threads = 8L;
    if (argc > 2) {
        num_threads = atol (argv[2]);
        if (num_threads < 1L) {
            num_threads = 8L;
        }
    }
    
    // Initialize global storage
    long i;
    globaltrap = (long double *)calloc(num_threads, sizeof(long double));
    globalmid = (long double *)calloc(num_threads, sizeof(long double));
    limits ** funct_args = (limits **)calloc(num_threads, sizeof(limits *));
    if (globaltrap == 0 || globalmid == 0 || funct_args == 0) {
        printf("Error allocating memory. Now exiting.\n");
        return -1;
    }
    for (i = 0L; i < num_threads; i++) {
        funct_args[(int)i] = (limits *)calloc(1, sizeof(limits));
        if (funct_args[(int)i] == 0) {
            printf("Error allocating memory. Now exiting.\n");
            return -1;
        }
    }
    pthread_t tid[(int)num_threads];
    
    // Split off worker threads. When dividing the work, if the number of 
    // threads does not evenly divide into the desired number of iterations,
    // give any extra iterations to the final thread. This gives the final
    // thread at most (num_threads - 1) extra iterations. 
    long clock_start = (long)clock();
    for (i = 0L; i < num_threads; i++) {
        funct_args[(int)i]->threadid = i;
        funct_args[(int)i]->lowlimit = i * (iterations / num_threads);
        funct_args[(int)i]->highlimit = (i + 1L == num_threads) ? iterations : 
            ((i + 1L) * (iterations / num_threads));
        funct_args[(int)i]->totaliterations = iterations;
        int w = pthread_create (&tid[(int)i], NULL, calculate, funct_args[(int)i]);
        if (w < 0) {
            printf ("Error creating thread. Now terminating.\n");
            return -2;
        }
    }
    
    // Wait for all the threads to return and check them
    for (i = 0L; i < num_threads; i++) {
        int y = pthread_join (tid[(int)i], NULL);
        if (y < 0) {
            printf ("Error waiting for thread. Now terminating.\n");
            return -3;
        }
    }
    
    // After worker threads end, clean up each of the partial sums
    long double mid = 0.0L;
    long double trap = 0.0L;
    for (i = 0L; i < num_threads; i++) {
        trap = trap + globaltrap[(int)i];
        mid = mid + globalmid[(int)i];
    }
    long clock_end = (long)clock();
    
    // Free global storage
    free(globaltrap);
    free(globalmid);
    for (i = 0L; i < num_threads; i++) {
        free(funct_args[(int)i]);
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
void * calculate (void * args) 
{
    // Initialize needed variables
    limits * funct_args = (limits *)args;
    long double trap = 0.0L, mid = 0.0L;
    long double inverseiterations = 1.0L / (long double)funct_args->totaliterations;
    long i;
    long k = funct_args->lowlimit;
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
    globaltrap[(int)(funct_args->threadid)] = trap;
    globalmid[(int)(funct_args->threadid)] = mid;
    pthread_exit (NULL);
}
