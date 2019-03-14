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
you input. You can also specify the number of bits to use in the precision
of the calculations. Obviously, the higher the number, the more digits 
you can successfully calculate. For best results, this number should be
a power of 2. To compile this code, run the following at the command line: 
    cc -O1 -Wall -c make_pi_3.c 
    cc -L~/gmpinstall -lgmp -lpthread -lrt -o make_pi_3 make_pi_3.o
    rm make_pi_3.o
Then to run it, just give it the iterations and threads arguments:
    make_pi_3 20000 8 512
Note that this uses the GNU Multiple Precision arithmetic library, found at 
http://gmplib.org/ and is only available for Unix/Linux. This library is a 
little less straightforward to use than p_thread. First, download it from 
the website and unzip it. Then, run ./configure and make. Finally, run
make install to install it to /usr/lib. If you're on a server like me and
can't install there, you'll need to pass the --prefix=/foldername option
to ./configure, and then pass that same folder to the -L option of gcc
when you build (otherwise, the -L option isn't necessary). Finally, make
sure you copy gmp.h into the same directory as the source code.
*/

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <gmp.h>

// Global results arrays
mpf_t * globaltrap;
mpf_t * globalmid;

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
    int bit_precision = 512;
    if (argc > 3) {
        bit_precision = atoi (argv[3]);
        if (bit_precision < 1) {
            bit_precision = 512;
        }
    }
    
    // Initialize global storage
    long i;
    char * accepted_pi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679\0";
    globaltrap = (mpf_t *)calloc(num_threads, sizeof(mpf_t));
    globalmid = (mpf_t *)calloc(num_threads, sizeof(mpf_t));
    limits ** funct_args = (limits **)calloc(num_threads, sizeof(limits *));
    if (globaltrap == 0 || globalmid == 0 || funct_args == 0) {
        printf("Error allocating memory. Now exiting.\n");
        return -1;
    }
    for (i = 0L; i < num_threads; i++) {
        mpf_init(globaltrap[(int)i]);
        mpf_init(globalmid[(int)i]);
    }
    for (i = 0L; i < num_threads; i++) {
        funct_args[(int)i] = (limits *)calloc(1, sizeof(limits));
        if (funct_args[(int)i] == 0) {
            printf("Error allocating memory. Now exiting.\n");
            return -1;
        }
    }
    pthread_t tid[(int)num_threads];
    mpf_set_default_prec(bit_precision);
    
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
    mpf_t trap;
    mpf_t mid;
    mpf_init(trap);
    mpf_init(mid);
    for (i = 0L; i < num_threads; i++) {
        mpf_add(trap, trap, globaltrap[(int)i]);
        mpf_add(mid, mid, globalmid[(int)i]);
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
    mpf_t simp;
    mpf_init(simp);
    mpf_mul_ui(mid, mid, 2L);
    mpf_add(trap, trap, mid);
    mpf_div_ui(trap, trap, 3L);
    mpf_mul_ui(simp, trap, 4L);
    gmp_printf("The calculated value of pi is %.*Ff\n", bit_precision/10, simp);
    gmp_printf("The actual value of pi is     %.*s\n", 
        (((bit_precision/10) > 100) ? 100 : ((bit_precision/10) + 2)), accepted_pi);
    printf("The time taken to calculate this was %.2f seconds\n", 
        ((float)(clock_end - clock_start)) / (float)CLOCKS_PER_SEC);
    mpf_clear(trap);
    mpf_clear(mid);
    mpf_clear(simp);
    return 0;
}

// Function executed by each thread to incrementally calculate the overall value
void * calculate (void * args) 
{
    // Initialize needed variables
    limits * funct_args = (limits *)args;
    mpf_t trap;
    mpf_t mid;
    mpf_t inverseiterations;
    mpf_init(trap);
    mpf_init(mid);
    mpf_init(inverseiterations);
    mpf_set_ui(inverseiterations, (funct_args->totaliterations));
    mpf_ui_div(inverseiterations, 1L, inverseiterations);
    long i;
    long k = funct_args->lowlimit;
    mpf_t temp_holder;
    mpf_init(temp_holder);
    mpf_set(temp_holder, inverseiterations);
    mpf_div_ui(temp_holder, temp_holder, 2L);
    mpf_t inc;
    mpf_init(inc);
    mpf_set_ui(inc, k);
    mpf_mul(inc, inc, inverseiterations);
    mpf_add(inc, inc, temp_holder);
    mpf_init(temp_holder);
    mpf_t leftrect;
    mpf_t rightrect;
    
    for (i = funct_args->lowlimit; i < funct_args->highlimit; i++) {
        // First, the trapezoid rule is used to estimate pi
        mpf_init(leftrect);
        mpf_set_ui(leftrect, k);
        mpf_mul(leftrect, leftrect, inverseiterations);
        k++;
        mpf_init(rightrect);
        mpf_set_ui(rightrect, k);
        mpf_mul(rightrect, rightrect, inverseiterations);
        mpf_add(temp_holder, leftrect, rightrect);
        mpf_div_ui(temp_holder, temp_holder, 2L);
        mpf_mul(temp_holder, temp_holder, temp_holder);
        mpf_add_ui(temp_holder, temp_holder, 1L);
        mpf_ui_div(temp_holder, 1L, temp_holder);
        mpf_mul(temp_holder, temp_holder, inverseiterations);
        mpf_add(trap, trap, temp_holder);
        mpf_init(temp_holder);
        
        // Next, the midpoint rule is also used to estimate pi
        mpf_set(temp_holder, inc);
        mpf_add(inc, inc, inverseiterations);
        mpf_mul(temp_holder, temp_holder, temp_holder);
        mpf_add_ui(temp_holder, temp_holder, 1L);
        mpf_ui_div(temp_holder, 1L, temp_holder);
        mpf_mul(temp_holder, temp_holder, inverseiterations);
        mpf_add(mid, mid, temp_holder);
        mpf_init(temp_holder);
    }
    
    // Save partial result and exit
    mpf_set(globaltrap[(int)(funct_args->threadid)], trap);
    mpf_set(globalmid[(int)(funct_args->threadid)], mid);
    mpf_clear(trap);
    mpf_clear(mid);
    mpf_clear(inverseiterations);
    mpf_clear(temp_holder);
    mpf_clear(inc);
    mpf_clear(leftrect);
    mpf_clear(rightrect);
    pthread_exit (NULL);
}
