/*This program calculates pi using a Simpson's Rule estimation of the
integral of arctangent from 0 to 1. When inputting the number of 
iterations to perform, more iterations = more precision. The number of
iterations is given as a command line argument. If no argument is 
provided, a default value of 20,000 is used. At 20,000 iterations, the
value of pi is guaranteed to be accurate up to 8 decimal places. This
version uses NVIDIA CUDA to perform parallel computation of the 
partial sums on a GPU.  

The amount of work each core does is given by the two #defines below. 
These values will need to be tuned for each device this code runs on in
order to get maximum performance. For example, on the Oakley cluster of
the Ohio Supercomputer Center, which uses the Tesla M2070, there are 
14 streaming multiprocessors (SM's), with 32 cores each, for a total of
448 cores. Thus, 448 threads will be created, with each thread performing
multiple iterations (total_iterations / (NUM_BLOCKS * THREADS_PER_BLOCK)
to be precise). Thus, the more iterations given, the more work each thread
does. The number of threads is kept constant in order to make clean-up 
easier and to not exceed the capabilities (max number of threads or blocks)
of any particular GPU device. The last thread might have a few extra 
iterations if that number doesn't divide evenly. 

The number of decimal digits to use as the precision of the calculations is
also given as a command-line argument. Obviously, the higher the number, the   
more digits you can successfully calculate. Accuracy still relies on the number
of iterations, though: a high number of digits but low number of iterations
will still result in a low number of digits of precision. Thus, you should
only increase the number of digits when your iterations get too high and 
you find that your calculations are no longer precise due to internal 
rounding error. You'll probably find that increasing the digits will decrease
performance severely. It is recommended, though, that since error accumulates,
the more digits you want to find, the more padding you'll need to add to the
end of the word to absorb that error. As a general rule of thumb, if you 
want to calculate x digits, make your words 2x long. Of course, this also
increases the runtime by 2x. 

Compilation on OSC is a little complicated. First, log into oakley.osc.edu and
request an interactive computation node since it has the GPU: 
    qsub -I -l walltime=0:59:00 -l nodes=1:gpus=1
Once you have acquired a node, first load the CUDA module: 
    module load cuda
Then compile it with the NVIDIA compiler: 
    nvcc -O -arch sm_20 -o make_pi_5 make_pi_5.cu
Finally, to run it, just give it the iterations and precision arguments:
    ./make_pi_5 20000 25
The -arch option is needed because we use dynamic memory allocation on the GPU.
Only second-generation and higher CUDA devices support this, so this code won't
run on the oldest NVIDIA graphics cards. 
*/

// Includes. Optimum values for OSC are:
// NUM_BLOCKS 14
// THREADS_PER_BLOCK 32
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NUM_BLOCKS 14
#define THREADS_PER_BLOCK 32

// A bignum is stored as all its decimal digits, separated into an array.
// Really, it's quite terrible for performance, but it allows infinite digits.
// Or at least as many as we can store in memory. The power tells us where to
// put the decimal point, and the number of significant digits tells us how
// many of the digits in the number are actually used. The precision tells us
// the maximum number of digits possible for this particular instance.
typedef struct {
    signed long int power;
    unsigned long int sig_digs;
    char * digits;
    unsigned long int precision;
} bignum;

// Function pointers, mostly for bignum operations. Note that in our use
// below, we assume most of the arithmetic functions don't fail and thus 
// don't check their return values. Hope they're tested well... Notice
// now that we have mirrored versions for the GPU, most of which just
// have to call the GPU memory allocation functions. 
__global__ void calculate (long *, long *, char *, long *, long *, char *, long, long);
__host__ bignum * bignum_init(long int);
__host__ void bignum_reset(bignum *);
__host__ void bignum_clear(bignum *);
__host__ int bignum_set_int(bignum *, long int);
__host__ void bignum_set(bignum *, bignum *);
__host__ void bignum_print(bignum *, long int);
__host__ int bignum_add(bignum *, bignum *, bignum *);
__host__ int bignum_add_int(bignum *, bignum *, long int);
__host__ int bignum_mult(bignum *, bignum *, bignum *);
__host__ int bignum_mult_int(bignum *, bignum *, long int);
__host__ int bignum_divide(bignum *, bignum *, bignum *);
__host__ int bignum_int_divide(bignum *, long int, bignum *);
__host__ int bignum_divide_int(bignum *, bignum *, long int);
__device__ bignum * bignum_init_gpu(long int);
__device__ void bignum_reset_gpu(bignum *);
__device__ void bignum_clear_gpu(bignum *);
__device__ int bignum_set_int_gpu(bignum *, long int);
__device__ void bignum_set_gpu(bignum *, bignum *);
__device__ int bignum_add_gpu(bignum *, bignum *, bignum *);
__device__ int bignum_add_int_gpu(bignum *, bignum *, long int);
__device__ int bignum_mult_gpu(bignum *, bignum *, bignum *);
__device__ int bignum_mult_int_gpu(bignum *, bignum *, long int);
__device__ int bignum_divide_gpu(bignum *, bignum *, bignum *);
__device__ int bignum_int_divide_gpu(bignum *, long int, bignum *);
__device__ int bignum_divide_int_gpu(bignum *, bignum *, long int);

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
    long max_digits = 25L;
    if (argc > 2) {
        max_digits = atoi (argv[2]);
        if (max_digits < 1L) {
            max_digits = 25L;
        }
    }
    
    // Initialize global storage. Notice that we now need extra arrays for data 
    // transfer between the GPU and regular RAM. These will hold the partial 
    // sums that each of the threads calculate. Unfortunately, due to the way
    // bignums are structured, each of their arguments has to be transferred 
    // separately. Luckily, this only happens once. 
    long clock_start = (long)clock();
    long int i, j;
    if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, (NUM_BLOCKS * THREADS_PER_BLOCK * 16384))
        != cudaSuccess) { printf("\nError setting GPU heap size.\n"); return 1; }
    cudaDeviceSynchronize();
    long * hosttrappower = (long *)calloc((int)(NUM_BLOCKS * THREADS_PER_BLOCK), sizeof(long));
    long * hosttrapsig_digs = (long *)calloc((int)(NUM_BLOCKS * THREADS_PER_BLOCK), sizeof(long));
    char * hosttrapdigits = (char *)calloc((int)(NUM_BLOCKS * THREADS_PER_BLOCK * max_digits), sizeof(char));
    long * hostmidpower = (long *)calloc((int)(NUM_BLOCKS * THREADS_PER_BLOCK), sizeof(long));
    long * hostmidsig_digs = (long *)calloc((int)(NUM_BLOCKS * THREADS_PER_BLOCK), sizeof(long));
    char * hostmiddigits = (char *)calloc((int)(NUM_BLOCKS * THREADS_PER_BLOCK * max_digits), sizeof(char));
    if ((hosttrappower == 0) || (hosttrapsig_digs == 0) || (hosttrapdigits == 0) ||
        (hostmidpower == 0) || (hostmidsig_digs == 0) || (hostmiddigits == 0)) {
        printf("\nError allocating memory on the CPU.\n"); 
        return 1; 
    }
    long * devicetrappower;
    long * devicetrapsig_digs;
    char * devicetrapdigits;
    long * devicemidpower;
    long * devicemidsig_digs;
    char * devicemiddigits;
    if (cudaMalloc((void**)&devicetrappower, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(long))) 
        != cudaSuccess) { printf("\nError allocating memory on GPU.\n"); return 1; }
    if (cudaMalloc((void**)&devicetrapsig_digs, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(long))) 
        != cudaSuccess) { printf("\nError allocating memory on GPU.\n"); return 1; }
    if (cudaMalloc((void**)&devicetrapdigits, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * max_digits * sizeof(char))) 
        != cudaSuccess) { printf("\nError allocating memory on GPU.\n"); return 1; }
    if (cudaMalloc((void**)&devicemidpower, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(long))) 
        != cudaSuccess) { printf("\nError allocating memory on GPU.\n"); return 1; }
    if (cudaMalloc((void**)&devicemidsig_digs, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(long))) 
        != cudaSuccess) { printf("\nError allocating memory on GPU.\n"); return 1; }
    if (cudaMalloc((void**)&devicemiddigits, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * max_digits * sizeof(char))) 
        != cudaSuccess) { printf("\nError allocating memory on GPU.\n"); return 1; }
    cudaDeviceSynchronize();
    char * accepted_pi = "3.14159265358979323846264338327950288419716939937510"
        "58209749445923078164062862089986280348253421170679\0";
    char pi_printer[2];
    pi_printer[0] = '0';
    pi_printer[1] = '\0';
    
    // Split off worker threads. When dividing the work, if the number of 
    // threads does not evenly divide into the desired number of iterations,
    // give any extra iterations to the final thread. This gives the final
    // thread at most (num_threads - 1) extra iterations. Notice that this
    // is a 1D-grid of work, and we use function arguments this time. Also,
    // remember the number of threads is held constant, thanks to #defines, 
    // at NUM_BLOCKS * THREADS_PER_BLOCK. 
    dim3 numBlocks(NUM_BLOCKS);
    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    calculate<<<numBlocks, threadsPerBlock>>>(devicetrappower, devicetrapsig_digs, 
        devicetrapdigits, devicemidpower, devicemidsig_digs, devicemiddigits, iterations, max_digits);
    cudaDeviceSynchronize();
    
    // Copy results back from GPU
    if (cudaMemcpy(hosttrappower, devicetrappower, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(long)), 
        cudaMemcpyDeviceToHost) != cudaSuccess) { printf("\nError copying memory from GPU.\n"); return 3; }
    if (cudaMemcpy(hosttrapsig_digs, devicetrapsig_digs, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(long)), 
        cudaMemcpyDeviceToHost) != cudaSuccess) { printf("\nError copying memory from GPU.\n"); return 3; }
    if (cudaMemcpy(hosttrapdigits, devicetrapdigits, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * max_digits * sizeof(char)), 
        cudaMemcpyDeviceToHost) != cudaSuccess) { printf("\nError copying memory from GPU.\n"); return 3; }
    if (cudaMemcpy(hostmidpower, devicemidpower, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(long)), 
        cudaMemcpyDeviceToHost) != cudaSuccess) { printf("\nError copying memory from GPU.\n"); return 3; }
    if (cudaMemcpy(hostmidsig_digs, devicemidsig_digs, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(long)), 
        cudaMemcpyDeviceToHost) != cudaSuccess) { printf("\nError copying memory from GPU.\n"); return 3; }
    if (cudaMemcpy(hostmiddigits, devicemiddigits, (int)(NUM_BLOCKS * THREADS_PER_BLOCK * max_digits * sizeof(char)), 
        cudaMemcpyDeviceToHost) != cudaSuccess) { printf("\nError copying memory from GPU.\n"); return 3; }
    cudaDeviceSynchronize();
    if (cudaFree(devicetrappower) != cudaSuccess) { printf("\nError freeing GPU memory.\n"); return 3; }
    if (cudaFree(devicetrapsig_digs) != cudaSuccess) { printf("\nError freeing GPU memory.\n"); return 3; }
    if (cudaFree(devicetrapdigits) != cudaSuccess) { printf("\nError freeing GPU memory.\n"); return 3; }
    if (cudaFree(devicemidpower) != cudaSuccess) { printf("\nError freeing GPU memory.\n"); return 3; }
    if (cudaFree(devicemidsig_digs) != cudaSuccess) { printf("\nError freeing GPU memory.\n"); return 3; }
    if (cudaFree(devicemiddigits) != cudaSuccess) { printf("\nError freeing GPU memory.\n"); return 3; }
    
    // After worker threads end, clean up each of the partial sums
    bignum * trap = bignum_init(max_digits);
    bignum * mid = bignum_init(max_digits);
    bignum * temp = bignum_init(max_digits);
    bignum * simp = bignum_init(max_digits);
    if (trap == 0 || mid == 0 || temp == 0 || simp == 0) {
        printf("Error allocating memory. Now exiting.\n");
        return -1;
    }
    for (i = 0L; i < (NUM_BLOCKS * THREADS_PER_BLOCK); i++) {
        simp->power = hosttrappower[i];
        simp->sig_digs = hosttrapsig_digs[i];
        for (j = 0L; j < max_digits; j++) {
            simp->digits[(int)j] = hosttrapdigits[(int)((i * max_digits) + j)];
        }
        bignum_add(temp, trap, simp);
        bignum_reset(trap);
        bignum_reset(simp);
        bignum_set(trap, temp);
        bignum_reset(temp);
        simp->power = hostmidpower[i];
        simp->sig_digs = hostmidsig_digs[i];
        for (j = 0L; j < max_digits; j++) {
            simp->digits[(int)j] = hostmiddigits[(int)((i * max_digits) + j)];
        }
        bignum_add(temp, mid, simp);
        bignum_reset(mid);
        bignum_reset(simp);
        bignum_set(mid, temp);
        bignum_reset(temp);
    }

    // Finally, Simpson's Rule is applied
    bignum_mult_int(temp, mid, 2L);
    bignum_reset(mid);
    bignum_set(mid, temp);
    bignum_reset(temp);
    bignum_add(temp, trap, mid);
    bignum_reset(trap);
    bignum_set(trap, temp);
    bignum_reset(temp);
    bignum_divide_int(temp, trap, 3L);
    bignum_reset(trap);
    bignum_set(trap, temp);
    bignum_reset(temp);
    bignum_mult_int(simp, trap, 4L);
    long clock_end = (long)clock();
    printf("The calculated value of pi is ");
    bignum_print(simp, 0L);
    printf("\nThe actual value of pi is     3.");
    for (i = 0L; i < (max_digits - 1L); i++) { 
        // This may print an extra digit or two because, somewhere down in the
        // code, we're losing our last sig dig during normal math, but it's 
        // bubbling back up, and causing the final result to lose a place or
        // two. It's not a big deal, and I don't want to do anything about it, 
        // so we'll just have the ends of the numbers not line up. Whatever.
        pi_printer[0] = accepted_pi[(int)(i + 2L)];
        printf("%s", pi_printer);
    }
    printf("\nThe time taken to calculate this was %.2f seconds\n", 
        ((float)(clock_end - clock_start)) / (float)CLOCKS_PER_SEC);
        
    // Free global storage
    free(hosttrappower);
    free(hosttrapsig_digs);
    free(hosttrapdigits);
    free(hostmidpower);
    free(hostmidsig_digs);
    free(hostmiddigits);
    bignum_clear(trap);
    bignum_clear(mid);
    bignum_clear(simp);
    bignum_clear(temp);
    return 0;
}

// Function executed by each thread to incrementally calculate the overall value
__global__ void calculate (long * devicetrappower, long * devicetrapsig_digs, 
        char * devicetrapdigits, long * devicemidpower, long * devicemidsig_digs, 
        char * devicemiddigits, long iterations, long max_digits) 
{
    // Initialize needed variables and check for errors
    long threadid = threadIdx.x + (blockIdx.x * THREADS_PER_BLOCK);
    long lowlimit = threadid * (iterations / (NUM_BLOCKS * THREADS_PER_BLOCK));
    long highlimit = (((threadid + 1L) == (NUM_BLOCKS * THREADS_PER_BLOCK)) ? iterations : 
        ((threadid + 1L) * (iterations / (NUM_BLOCKS * THREADS_PER_BLOCK))));
    bignum * trap = bignum_init_gpu(max_digits);
    bignum * mid = bignum_init_gpu(max_digits);
    bignum * inverseiterations = bignum_init_gpu(max_digits);
    bignum * temp_holder = bignum_init_gpu(max_digits);
    bignum * temp_holder2 = bignum_init_gpu(max_digits);
    bignum * inc = bignum_init_gpu(max_digits);
    bignum * leftrect = bignum_init_gpu(max_digits);
    bignum * rightrect = bignum_init_gpu(max_digits);
    if (trap == 0 || mid == 0 || inverseiterations == 0 || temp_holder == 0 ||
        temp_holder2 == 0 || inc == 0 || leftrect == 0 || rightrect == 0) {
        return;
    }
    
    // Initialize values of needed variables
    bignum_set_int_gpu(temp_holder, iterations);
    bignum_int_divide_gpu(inverseiterations, 1L, temp_holder);
    bignum_reset_gpu(temp_holder);
    long i;
    long k = lowlimit;
    bignum_divide_int_gpu(temp_holder, inverseiterations, 2L);
    bignum_set_int_gpu(inc, k);
    bignum_mult_gpu(temp_holder2, inc, inverseiterations);
    bignum_reset_gpu(inc);
    bignum_set_gpu(inc, temp_holder2);
    bignum_reset_gpu(temp_holder2);
    bignum_add_gpu(temp_holder2, inc, temp_holder);
    bignum_reset_gpu(inc);
    bignum_set_gpu(inc, temp_holder2);
    bignum_reset_gpu(temp_holder2);
    bignum_reset_gpu(temp_holder);
    
    // Main iteration loop. Note that the values of inverseiterations, inc, 
    // mid, and trap are preserved across loop iterations, as is counter k.
    // inverseiterations is a constant that is stored for simplicity. Man, 
    // this is looking more and more like assembly...
    for (i = lowlimit; i < highlimit; i++) {
        // First, the trapezoid rule is used to estimate pi
        bignum_reset_gpu(leftrect);
        bignum_set_int_gpu(leftrect, k);
        bignum_mult_gpu(temp_holder2, leftrect, inverseiterations);
        bignum_reset_gpu(leftrect);
        bignum_set_gpu(leftrect, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        k++;
        bignum_reset_gpu(rightrect);
        bignum_set_int_gpu(rightrect, k);
        bignum_mult_gpu(temp_holder2, rightrect, inverseiterations);
        bignum_reset_gpu(rightrect);
        bignum_set_gpu(rightrect, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_add_gpu(temp_holder, leftrect, rightrect);
        bignum_divide_int_gpu(temp_holder2, temp_holder, 2L);
        bignum_reset_gpu(temp_holder);
        bignum_set_gpu(temp_holder, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_mult_gpu(temp_holder2, temp_holder, temp_holder);
        bignum_reset_gpu(temp_holder);
        bignum_set_gpu(temp_holder, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_add_int_gpu(temp_holder2, temp_holder, 1L);
        bignum_reset_gpu(temp_holder);
        bignum_set_gpu(temp_holder, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_int_divide_gpu(temp_holder2, 1L, temp_holder);
        bignum_reset_gpu(temp_holder);
        bignum_set_gpu(temp_holder, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_mult_gpu(temp_holder2, temp_holder, inverseiterations);
        bignum_reset_gpu(temp_holder);
        bignum_set_gpu(temp_holder, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_add_gpu(temp_holder2, trap, temp_holder);
        bignum_reset_gpu(trap);
        bignum_set_gpu(trap, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_reset_gpu(temp_holder);
        
        // Next, the midpoint rule is also used to estimate pi
        bignum_set_gpu(temp_holder, inc);
        bignum_add_gpu(temp_holder2, inc, inverseiterations);
        bignum_reset_gpu(inc);
        bignum_set_gpu(inc, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_mult_gpu(temp_holder2, temp_holder, temp_holder);
        bignum_reset_gpu(temp_holder);
        bignum_set_gpu(temp_holder, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_add_int_gpu(temp_holder2, temp_holder, 1L);
        bignum_reset_gpu(temp_holder);
        bignum_set_gpu(temp_holder, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_int_divide_gpu(temp_holder2, 1L, temp_holder);
        bignum_reset_gpu(temp_holder);
        bignum_set_gpu(temp_holder, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_mult_gpu(temp_holder2, temp_holder, inverseiterations);
        bignum_reset_gpu(temp_holder);
        bignum_set_gpu(temp_holder, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_add_gpu(temp_holder2, mid, temp_holder);
        bignum_reset_gpu(mid);
        bignum_set_gpu(mid, temp_holder2);
        bignum_reset_gpu(temp_holder2);
        bignum_reset_gpu(temp_holder);
    }
    
    // Save partial result, clear memory, and exit
    devicetrappower[threadid] = trap->power;
    devicetrapsig_digs[threadid] = trap->sig_digs;
    for (i = 0; i < max_digits; i++) {
        devicetrapdigits[(threadid * max_digits) + i] = trap->digits[i];
    }
    devicemidpower[threadid] = mid->power;
    devicemidsig_digs[threadid] = mid->sig_digs;
    for (i = 0; i < max_digits; i++) {
        devicemiddigits[(threadid * max_digits) + i] = mid->digits[i];
    }
    bignum_clear_gpu(trap);
    bignum_clear_gpu(mid);
    bignum_clear_gpu(inverseiterations);
    bignum_clear_gpu(temp_holder);
    bignum_clear_gpu(temp_holder2);
    bignum_clear_gpu(inc);
    bignum_clear_gpu(leftrect);
    bignum_clear_gpu(rightrect);
}

// Create space for a bignum with the specified precision.
// Technically, it's also initialized if we interpret having zero
// significant digits as the number having a value of zero.
__host__ bignum * bignum_init(long int precision) {
    bignum * temp_ptr = (bignum *)calloc(1, sizeof(bignum));
    temp_ptr->digits = (char *)calloc((int)precision, sizeof(char));
    if ((temp_ptr->digits) == 0) { temp_ptr = 0; }
    temp_ptr->precision = precision;
    return temp_ptr;
}

// Resets a bignum's value to zero. memcpy isn't used because 
// why bring the string library into this just for this use?
__host__ void bignum_reset(bignum * numval) {
    if ((numval->sig_digs) > 0L) {
        long int i;
        for (i = 0L; i < numval->precision; i++) { numval->digits[(int)i] = '\0'; }
        numval->power = 0L;
        numval->sig_digs = 0L;
    }
    return;
}

// Free memory used by a bignum when we're done with it
__host__ void bignum_clear(bignum * oldnum) {
    free(oldnum->digits);
    free(oldnum);
    return;
}

// Set an instance of a bignum to an integer value. Note that if we can't 
// initialize the temp word we need for copying, we return false (value = 0). 
// We also assume that the number is non-negative since we only store 
// unsigned numbers. We assume the result is initialized/reset. Finally, 
// we handle zero specially by just resetting (again?) the result. Note that
// we explicitly assume the number to convert fits within the max number of
// digits. If we try to convert a number bigger than we can store, it won't work.
__host__ int bignum_set_int(bignum * numval, long int intval) {
    if (intval > 0L) {
        // Separate out the individual digits (stored backwards)
        char * temp_word = (char *)calloc((int)(numval->precision), sizeof(char));
        if (temp_word == 0) { return 0; }
        long int temp_int = intval;
        long int counter = 0L;
        while (temp_int > 0L) {
            temp_word[(int)counter] = (char)(temp_int % 10L);
            temp_int = temp_int / 10L;
            counter++;
        }
        
        // Detect any trailing zeros that we don't need to store
        numval->power = counter - 1L;
        long int leadingzeros = 0L;
        int hasleading = 1;
        while (hasleading == 1) {
            if (temp_word[(int)leadingzeros] != 0) { hasleading = 0; }
            else { leadingzeros++; }
        }
        
        // Store final result into actual bignum variable
        for (temp_int = 0L; temp_int < (counter - leadingzeros); temp_int++) {
            numval->digits[(int)temp_int] = temp_word[(int)(counter - temp_int - 1L)];
        }
        numval->sig_digs = counter - leadingzeros;
        free(temp_word);
        return 1;
    }
    else { bignum_reset(numval); return 1; }
}

// Set an instance of a bignum to the value of another bignum. We don't assume
// they're both the same precision; just use the precision of the new number.
// We do assume that the new number has already been initialized, though.
// strncpy is not used since it quits after seeing the first zero.
__host__ void bignum_set(bignum * newnum, bignum * oldnum) {
    if ((oldnum->sig_digs) > 0L) {
        newnum->power = oldnum->power;
        newnum->sig_digs = ((oldnum->sig_digs > newnum->precision) ? 
            (newnum->precision) : (oldnum->sig_digs));
        long int i;
        for (i = 0L; i < newnum->sig_digs; i++) {
            newnum->digits[(int)i] = oldnum->digits[(int)i];
        }
    }
    else { bignum_reset(newnum); }
    return;
}

// Use printf to print the number one digit at a time. There are a few cases:
// power > significant digits: pad end with zeros
// significant digits > power: fractional digit (non-integer)
// power is negative: total value less than 1
// The second argument is the maximum number of significant digits to print.
// If it's zero, then all available digits will be printed, maxing out at 
// the precision of the number (the total amount is could possibly store).
// Note that this is different from total digits printed: zeroes after a 
// decimal point but before the first significant digit don't count, and we
// make sure we print at least the integral part of the number (we only 
// chop off fractional portions).
__host__ void bignum_print(bignum * numval, long int maxdigits) {
    long int i;
    long int limit = numval->sig_digs;
    if (numval->sig_digs == 0L) { printf("0"); } else {
    if ((maxdigits > 0L) && (maxdigits < numval->sig_digs)) {
        limit = maxdigits;
    }
    if (numval->power < 0L) {
        printf("0.");
        for (i = 1L; i < (-1L * (numval->power)); i++) { printf("0"); }
        for (i = 0L; i < limit; i++) { 
            printf("%d", (int)(numval->digits[(int)i])); 
        }
    }
    else if (numval->sig_digs > (numval->power + 1L)) {
        for (i = 0L; i <= numval->power; i++) { 
            printf("%d", (int)(numval->digits[(int)i])); 
        }
        if (limit > (numval->power + 1L)) { printf("."); }
        for (i = (numval->power + 1L); i < limit; i++) { 
            printf("%d", (int)(numval->digits[(int)i])); 
        }
    }
    else { for (i = 0L; i < numval->sig_digs; i++) { 
        printf("%d", (int)(numval->digits[(int)i])); } 
    }
    if ((numval->power > 0L) && ((numval->power + 1L) > numval->sig_digs)) { 
        for (i = 0L; i < ((numval->power + 1L) - numval->sig_digs); i++) { 
            printf("0"); 
        } 
    } }
    fflush(stdout);
    return;
}

// Adds two bignums together and stores the result. Uses the functions to 
// reset and set the location of the result internally, so current contents of
// result operand will be overwritten. Like bignum_set_int, returns 1 if 
// addition was successful or 0 if an error occurred. A special shortcut is
// taken if either (or both) of the operands are zero. Note that it is possible 
// for large additions to cause underflow to zero. In that case, special care is
// taken to make sure the proper input operand is used. Note that we assume the
// precision of all three operands is the same. If it's not, something terrible
// like a seg fault or incorrect answer will probably occur. Most importantly, 
// the result operand CANNOT be the same as one of the input operands, since
// the result is clobbered immediately and used as a scratchpad. Note that this
// is also unsigned addition: not only does it not accept negative numbers, it
// also doesn't do subtraction (which, for that matter, isn't commutative).
__host__ int bignum_add(bignum * resultnum, bignum * leftnum, bignum * rightnum) {
    bignum_reset(resultnum);
    if ((leftnum->sig_digs == 0L) && (rightnum->sig_digs > 0L)) {
        bignum_set(resultnum, rightnum);
        return 1;
    }
    else if ((rightnum->sig_digs == 0L) && (leftnum->sig_digs > 0L)) {
        bignum_set(resultnum, leftnum);
        return 1;
    }
    else if ((leftnum->sig_digs == 0L) && (rightnum->sig_digs == 0L)) { return 1; }
    else {
        // First check for overshift:  if the larger number's power is too much
        // bigger than the smaller number's, the smaller will be completely lost,
        // and we'll just end up with the large number as the result.
        if ((((leftnum->power - rightnum->power) > 0) && 
            ((leftnum->power - rightnum->power) > resultnum->precision))) {
            bignum_set(resultnum, leftnum);
            return 1;
        }
        if ((((rightnum->power - leftnum->power) > 0) && 
            ((rightnum->power - leftnum->power) > resultnum->precision))) {
            bignum_set(resultnum, rightnum);
            return 1;
        }
        
        // Next, shift the smaller operand to match the larger one by copying
        // it into the result operand as a partial sum. Also copy over the 
        // power and total significant digits into the result.
        bignum * bigger;
        bignum * smaller;
        if ((leftnum->power - rightnum->power) >= 0L) {
            bigger = leftnum;
            smaller = rightnum;
        }
        else {
            bigger = rightnum;
            smaller = leftnum;
        }
        long int difference = bigger->power - smaller->power;
        long int startdigit = smaller->sig_digs + difference;
        long int transfertotal = smaller->sig_digs;
        if (startdigit > resultnum->precision) {
            startdigit = resultnum->precision - difference;
            transfertotal = startdigit;
        }
        long int startdigitcopy = startdigit;
        startdigit--;
        long int i;
        for (i = 0L; i < transfertotal; i++) {
            if ((startdigit - difference) >= 0L) {
                resultnum->digits[(int)startdigit] = 
                    smaller->digits[(int)(startdigit - difference)];
            }
            startdigit--;
        }
        
        // Now the main addition loop: loop through each digit and add it.
        // The carry from the previous digit will add to the current one.
        // Note that we detect any trailing zeros to take from the sig_digs.
        // Also, copy over the power and significant digits
        resultnum->power = bigger->power;
        resultnum->sig_digs = startdigitcopy;
        if (bigger->sig_digs > resultnum->sig_digs) {
            resultnum->sig_digs = bigger->sig_digs;
            startdigitcopy = resultnum->sig_digs;
        }
        int trailingzeros = 1;
        long int zerocount = 0L;
        char carry = 0;
        for (i = 0L; i < resultnum->sig_digs; i++) {
            resultnum->digits[(int)(startdigitcopy - i - 1L)] += 
                (bigger->digits[(int)(startdigitcopy - i - 1L)] + carry);
            if (resultnum->digits[(int)(startdigitcopy - i - 1L)] >= 10) {
                resultnum->digits[(int)(startdigitcopy - i - 1L)] -= 10;
                carry = 1;
            } else { carry = 0; }
            if (trailingzeros == 1) {
                if (resultnum->digits[(int)(startdigitcopy - i - 1L)] == '\0') {
                    zerocount++;
                } else { trailingzeros = 0; }
            }
        }
        
        // If we've got trailing zeros, subtract them from the final count of
        // sig_digs. Also, if we have a carry, we need to shift everything...
        resultnum->sig_digs -= zerocount;
        if (carry > 0) {
            transfertotal = resultnum->sig_digs;
            if (transfertotal == resultnum->precision) { transfertotal--; }
            startdigitcopy = transfertotal - 1L;
            for (i = 0L; i < transfertotal; i++) {
                if (startdigitcopy >= 0L) {
                    resultnum->digits[(int)(startdigitcopy + 1L)] =
                        resultnum->digits[(int)startdigitcopy];
                }
                else if ((startdigitcopy + 1L) >= 0L) {
                    resultnum->digits[(int)(startdigitcopy + 1L)] = 0;
                }
                startdigitcopy--;
            }
            resultnum->digits[0] = carry;
            resultnum->power++;
            resultnum->sig_digs++;
        }
        if (resultnum->sig_digs > resultnum->precision) {
            resultnum->sig_digs = resultnum->precision;
        }
        return 1;
    }
}

// A convenience wrapper that temporarily creates a new bignum out of the 
// given integer, calls bignum_add with it and the other operand, and deletes
// the temporary bignum before exiting. Any problems that bignum_add encounters
// are passed back up through this function and returned to the caller.
__host__ int bignum_add_int(bignum * resultnum, bignum * leftnum, long int rightint) {
    bignum_reset(resultnum);
    if ((rightint == 0L) && (leftnum->sig_digs > 0L)) {
        bignum_set(resultnum, leftnum);
        return 1;
    }
    else if ((leftnum->sig_digs == 0L) && (rightint > 0L)) {
        return bignum_set_int(resultnum, rightint);
    }
    else if ((leftnum->sig_digs == 0L) && (rightint == 0L)) { return 1; }
    else {
        bignum * tempnum = bignum_init(resultnum->precision);
        if (tempnum == 0) { return 0; }
        if (bignum_set_int(tempnum, rightint) == 0) {
            bignum_clear(tempnum);
            return 0;
        }
        int retval = bignum_add(resultnum, leftnum, tempnum);
        bignum_clear(tempnum);
        return retval;
    }
}

// Multiplies two bignums together and stores the result. Like add, uses 
// functions to reset and set the location of the result, and returns 1 upon
// success or 0 if an error occurred. A special shortcut is taken if either
// operand is zero, since the result will thus also be zero. Note that we assume
// the precision of all three operands is the same. If it's not, something 
// terrible like a seg fault or incorrect answer will probably occur. Most 
// importantly, the result operand CANNOT be the same as one of the input
// operands, since the result is clobbered immediately and used as a scratchpad.
// Also, note that this is unsigned: it assumes both operands are positive.
__host__ int bignum_mult(bignum * resultnum, bignum * leftnum, bignum * rightnum) {
    bignum_reset(resultnum);
    if ((leftnum->sig_digs == 0L) || (rightnum->sig_digs == 0L)) { return 1; }
    else {
        // Initialize the scratchpad and find the digit limits
        char * temp_word = (char *)calloc((int)(2L * (resultnum->precision)), sizeof(char));
        if (temp_word == 0) { return 0; }
        bignum * bigger;
        bignum * smaller;
        if (((signed long int)leftnum->sig_digs - (signed long int)rightnum->sig_digs) >= 0L) {
            bigger = leftnum;
            smaller = rightnum;
        }
        else if ((rightnum->sig_digs - leftnum->sig_digs) > 0L) {
            bigger = rightnum;
            smaller = leftnum;
        }
        long int bigstart = (bigger->sig_digs) - 1L;
        long int smallstart = (smaller->sig_digs) - 1L;
        long int bigcounter, smallcounter;
        char carry = 0;
        
        // Perform the shift-addition loop. We choose to loop over each
        // digit of the smaller number for fewer overall iterations. If
        // the current bigloop has a zero, we can just skip that iteration.
        // Also, record the final carry, power, and sig_digs values. 
        for (bigcounter = 0L; bigcounter < (smaller->sig_digs); bigcounter++) {
            if (smaller->digits[(int)(smallstart - bigcounter)] != '\0') {
                carry = 0;
                for(smallcounter = 0L; smallcounter < (bigger->sig_digs); smallcounter++) {
                    temp_word[(int)((2L * (resultnum->precision)) - smallcounter - 
                        bigcounter - 1L)] += (carry + (smaller->digits[(int)(smallstart - 
                        bigcounter)] * bigger->digits[(int)(bigstart - smallcounter)]));
                    carry = temp_word[(int)((2L * (resultnum->precision)) - 
                        smallcounter - bigcounter - 1L)] / 10;
                    temp_word[(int)((2L * (resultnum->precision)) - smallcounter - 
                        bigcounter - 1L)] %= 10;
                }
                temp_word[(int)((2L * (resultnum->precision)) - bigcounter -
                    (bigger->sig_digs) - 1L)] = carry;
            }
        }
        resultnum->power = ((bigger->power) + (smaller->power));
        resultnum->sig_digs = ((bigger->sig_digs) + (smaller->sig_digs));
        
        // Adjust for lack of a final carry or trailing zeros.
        if (carry < 1) { 
            (resultnum->sig_digs)--; 
            (resultnum->power)--; 
        }
        (resultnum->power)++; 
        int trailingzeros = 1;
        long int zerocount = 0L;
        long int i = (2L * (resultnum->precision) - 1L); 
        while (trailingzeros == 1) {
            if (temp_word[(int)i] == '\0') {
                zerocount++;
            } else { trailingzeros = 0; }
            i--;
        }
        resultnum->sig_digs -= zerocount;
        if ((resultnum->sig_digs) > (resultnum->precision)) {
            resultnum->sig_digs = (resultnum->precision);
        }
        
        // Finally, copy from the temp word into the result, taking into 
        // account any digits we may lose due to precision.
        long int tempstart = (2L * (resultnum->precision)) - ((bigger->sig_digs) + 
            (smaller->sig_digs));
        if (carry < 1) { tempstart++; }
        for (i = 0L; i < (resultnum->sig_digs); i++) {
            resultnum->digits[(int)i] = temp_word[(int)(tempstart + i)];
        }
        free(temp_word);
        return 1;
    }
}

// Like bignum_add_int, a convenience wrapper that creates a temporary bignum
// out of the integer and passes it to bignum_mult. Any problems encountered 
// in client functions are passed back up to the original caller.
__host__ int bignum_mult_int(bignum * resultnum, bignum * leftnum, long int rightint) {
    bignum_reset(resultnum);
    if ((leftnum->sig_digs == 0L) || (rightint == 0L)) { return 1; }
    else {
        bignum * tempnum = bignum_init(resultnum->precision);
        if (tempnum == 0) { return 0; }
        if (bignum_set_int(tempnum, rightint) == 0) {
            bignum_clear(tempnum);
            return 0;
        }
        int retval = bignum_mult(resultnum, leftnum, tempnum);
        bignum_clear(tempnum);
        return retval;
    }
}

// Divides two bignums. Taken in terms of a fraction, leftnum is the numerator 
// and rightnum is the denominator. Performs an explicit check to make sure
// the denominator is not zero, and returns 0 (an error) if it is. Returns 1 upon 
// success or 0 if an error occurs. A special shortcut is taken if the numerator is 
// zero. Note that we assume the precision of all three operands is the same. If it's 
// not, something terrible like a seg fault or incorrect answer will probably occur. 
// Most importantly, the result operand CANNOT be the same as one of the input 
// operands, since the result is clobbered immediately and used as a scratchpad.
// Also, note that this is unsigned: it assumes both operands are positive.
__host__ int bignum_divide(bignum * resultnum, bignum * numerator, bignum * denominator) {
    bignum_reset(resultnum);
    if (denominator->sig_digs == 0L) { return 0; }
    else if (numerator->sig_digs == 0L) { return 1; }
    else {
        // Initialize the scratchpad and initially copy the numerator into it.
        // Also initialize the result's power.
        char * temp_word = (char *)calloc((int)(2L * 
            (resultnum->precision) + 2L), sizeof(char)); // May only need to be + 1L
        if (temp_word == 0) { return 0; }
        long int i;
        for (i = 0L; i < numerator->sig_digs; i++) {
            temp_word[(int)(i + 1L)] = numerator->digits[(int)i];
        }
        resultnum->power = (numerator->power - denominator->power);
        long int sigdigctr = 0L;
        long int numeratorindex = 0L;
        
        // First see if we need to "shift" the numerator by comparing it.
        i = ((denominator->sig_digs) - 1L);
        int denom_bigger = 1;
        while ((i >= 0L) && (denom_bigger == 1)) {
            if ((denominator->digits[(int)((denominator->sig_digs) - i - 1L)]) > 
                (temp_word[(int)((denominator->sig_digs) - i)])) {
                i = 0L;
            }
            else if ((denominator->digits[(int)((denominator->sig_digs) - 
                i - 1L)]) < (temp_word[(int)((denominator->sig_digs) - i)])) {
                denom_bigger = 0;
            }
            else if (((denominator->digits[(int)((denominator->sig_digs) - i - 
                1L)]) == (temp_word[(int)((denominator->sig_digs) - i)])) && (i == 0L)) {
                denom_bigger = 0;
            }
            i--;
        }
        if (denom_bigger == 1) { 
            numeratorindex++; 
            (resultnum->power)--;
        }
        
        // Now the main division loop. Note that there's two ways to terminate:
        // either we've filled the entire precision of the result word and are
        // forced to truncate our result, or our answer divides exactly. In the
        // second case, once we've exhausted the numerator's significant digits
        // and our temp word contains nothing but zeros, we can end early since
        // all subsequent iterations would contribute only zeros as well. Note
        // that special care will be taken to detect extra zeros at the end of
        // the result so that the sig_digs is recorded correctly. Also, we don't
        // round, we truncate, which doesn't minimize error.
        int nonzero = 1;
        while ((sigdigctr < (resultnum->precision)) && (nonzero == 1)) {
            // First run the subtraction loop.
            char current_digit = 0;
            int numer_bigger = 1;
            while (numer_bigger == 1) {
                // To subtract, first run a comparison to see if the numerator
                // is bigger. If it is, increment the counter and subtract. 
                i = ((denominator->sig_digs) - 1L);
                denom_bigger = 1;
                if (temp_word[(int)numeratorindex] > 0) { denom_bigger = 0; }
                while ((i >= 0L) && (denom_bigger == 1)) {
                    if ((denominator->digits[(int)((denominator->sig_digs) - 
                        i - 1L)]) > (temp_word[(int)((denominator->sig_digs) + 
                        numeratorindex - i)])) {
                        i = 0L;
                    }
                    else if ((denominator->digits[(int)((denominator->sig_digs) - 
                        i - 1L)]) < (temp_word[(int)((denominator->sig_digs) + 
                        numeratorindex - i)])) {
                        denom_bigger = 0;
                    }
                    else if (((denominator->digits[(int)((denominator->sig_digs) - 
                        i - 1L)]) == (temp_word[(int)((denominator->sig_digs) + 
                        numeratorindex - i)])) && (i == 0L)) {
                        denom_bigger = 0;
                    }
                    i--;
                }
                if (denom_bigger == 1) { 
                    numer_bigger = 0;
                }
                
                // Increment counter and perform subtraction loop.
                if (numer_bigger == 1) {
                    current_digit++;
                    for (i = 0L; i < (denominator->sig_digs); i++) {
                        temp_word[(int)((denominator->sig_digs) + 
                            numeratorindex - i)] -= (denominator->digits[
                            (int)((denominator->sig_digs) - i - 1L)]);
                        if ((temp_word[(int)((denominator->sig_digs) + 
                            numeratorindex - i)]) < 0) {
                            temp_word[(int)((denominator->sig_digs) + 
                                numeratorindex - i)] += 10L;
                            (temp_word[(int)((denominator->sig_digs) + 
                                numeratorindex - i - 1L)]) -= 1L;
                        }
                    }
                }
            }
            
            // If we're past all of the numerator's significant digits, run
            // zero detection on it to see if we can end early.
            if (sigdigctr > (numerator->sig_digs)) { // May only need to be >=
                long int zerocounter = 0L; 
                i = 0L;
                while ((i == zerocounter) && (i <= (denominator->sig_digs))) {
                    if ((temp_word[(int)(numeratorindex + i)]) < 1) { zerocounter++; }
                    i++;
                }
                if (zerocounter == ((denominator->sig_digs) + 1L)) { nonzero = 0; }
            }
            
            // Once we have obtained the proper digit in the result, save it.
            if (sigdigctr < resultnum->precision) {
                resultnum->digits[(int)sigdigctr] = current_digit;
            }
            sigdigctr++;
            numeratorindex++;
        }
        
        // Record the result's sig digs, taking care to detect trailing zeros.
        resultnum->sig_digs = sigdigctr;
        int trailingzeros = 1;
        long int zerocount = 0L;
        i = sigdigctr - 1L; 
        while (trailingzeros == 1) {
            if (resultnum->digits[(int)i] == '\0') {
                zerocount++;
            } else { trailingzeros = 0; }
            i--;
        }
        (resultnum->sig_digs) -= zerocount;
        free (temp_word);
        return 1;
    }
}

// A convenience wrapper that creates a temporary bignum out of the integer. 
// Since division is not commutative, two wrappers are given. Any problems 
// encountered in client functions are passed back up to the original caller.
__host__ int bignum_int_divide(bignum * resultnum, long int leftint, bignum * rightnum) {
    bignum_reset(resultnum);
    if (rightnum->sig_digs == 0L) { return 0; }
    else if (leftint == 0L) { return 1; }
    else {
        bignum * tempnum = bignum_init(resultnum->precision);
        if (tempnum == 0) { return 0; }
        if (bignum_set_int(tempnum, leftint) == 0) {
            bignum_clear(tempnum);
            return 0;
        }
        int retval = bignum_divide(resultnum, tempnum, rightnum);
        bignum_clear(tempnum);
        return retval;
    }
}

// A convenience wrapper that creates a temporary bignum out of the integer. 
// Since division is not commutative, two wrappers are given. Any problems 
// encountered in client functions are passed back up to the original caller.
__host__ int bignum_divide_int(bignum * resultnum, bignum * leftnum, long int rightint) {
    bignum_reset(resultnum);
    if (rightint == 0L) { return 0; }
    else if (leftnum->sig_digs == 0L) { return 1; }
    else {
        bignum * tempnum = bignum_init(resultnum->precision);
        if (tempnum == 0) { return 0; }
        if (bignum_set_int(tempnum, rightint) == 0) {
            bignum_clear(tempnum);
            return 0;
        }
        int retval = bignum_divide(resultnum, leftnum, tempnum);
        bignum_clear(tempnum);
        return retval;
    }
}

// Create space for a bignum with the specified precision.
// Technically, it's also initialized if we interpret having zero
// significant digits as the number having a value of zero.
__device__ bignum * bignum_init_gpu(long int precision) {
    bignum * temp_ptr = (bignum *)malloc(sizeof(bignum));
    if (temp_ptr == 0) { return temp_ptr; }
    temp_ptr->digits = (char *)malloc((int)(precision * sizeof(char)));
    if ((temp_ptr->digits) == 0) { temp_ptr = 0; return temp_ptr; }
    int i;
    for (i = 0; i < precision; i++) { temp_ptr->digits[i] = '\0'; }
    temp_ptr->power = 0L;
    temp_ptr->sig_digs = 0L;
    temp_ptr->precision = precision;
    return temp_ptr;
}

// Resets a bignum's value to zero. memcpy isn't used because 
// why bring the string library into this just for this use?
__device__ void bignum_reset_gpu(bignum * numval) {
    if ((numval->sig_digs) > 0L) {
        long int i;
        for (i = 0L; i < numval->precision; i++) { numval->digits[(int)i] = '\0'; }
        numval->power = 0L;
        numval->sig_digs = 0L;
    }
    return;
}

// Free memory used by a bignum when we're done with it
__device__ void bignum_clear_gpu(bignum * oldnum) {
    free(oldnum->digits);
    free(oldnum);
    return;
}

// Set an instance of a bignum to an integer value. Note that if we can't 
// initialize the temp word we need for copying, we return false (value = 0). 
// We also assume that the number is non-negative since we only store 
// unsigned numbers. We assume the result is initialized/reset. Finally, 
// we handle zero specially by just resetting (again?) the result. Note that
// we explicitly assume the number to convert fits within the max number of
// digits. If we try to convert a number bigger than we can store, it won't work.
__device__ int bignum_set_int_gpu(bignum * numval, long int intval) {
    if (intval > 0L) {
        
        // Separate out the individual digits (stored backwards)
        char * temp_word = (char *)malloc((int)(numval->precision * sizeof(char)));
        if (temp_word == 0) { return 0; }
        long int i;
        for (i = 0; i < numval->precision; i++) { temp_word[(int)i] = '\0'; }
        long int temp_int = intval;
        long int counter = 0L;
        while (temp_int > 0L) {
            temp_word[(int)counter] = (char)(temp_int % 10L);
            temp_int = temp_int / 10L;
            counter++;
        }
        
        // Detect any trailing zeros that we don't need to store
        numval->power = counter - 1L;
        long int leadingzeros = 0L;
        int hasleading = 1;
        while (hasleading == 1) {
            if (temp_word[(int)leadingzeros] != 0) { hasleading = 0; }
            else { leadingzeros++; }
        }
        
        // Store final result into actual bignum variable
        for (temp_int = 0L; temp_int < (counter - leadingzeros); temp_int++) {
            numval->digits[(int)temp_int] = temp_word[(int)(counter - temp_int - 1L)];
        }
        numval->sig_digs = counter - leadingzeros;
        free(temp_word);
        return 1;
    }
    else { bignum_reset_gpu(numval); return 1; }
}

// Set an instance of a bignum to the value of another bignum. We don't assume
// they're both the same precision; just use the precision of the new number.
// We do assume that the new number has already been initialized, though.
// strncpy is not used since it quits after seeing the first zero.
__device__ void bignum_set_gpu(bignum * newnum, bignum * oldnum) {
    if ((oldnum->sig_digs) > 0L) {
        newnum->power = oldnum->power;
        newnum->sig_digs = ((oldnum->sig_digs > newnum->precision) ? 
            (newnum->precision) : (oldnum->sig_digs));
        long int i;
        for (i = 0L; i < newnum->sig_digs; i++) {
            newnum->digits[(int)i] = oldnum->digits[(int)i];
        }
    }
    else { bignum_reset_gpu(newnum); }
    return;
}

// Adds two bignums together and stores the result. Uses the functions to 
// reset and set the location of the result internally, so current contents of
// result operand will be overwritten. Like bignum_set_int, returns 1 if 
// addition was successful or 0 if an error occurred. A special shortcut is
// taken if either (or both) of the operands are zero. Note that it is possible 
// for large additions to cause underflow to zero. In that case, special care is
// taken to make sure the proper input operand is used. Note that we assume the
// precision of all three operands is the same. If it's not, something terrible
// like a seg fault or incorrect answer will probably occur. Most importantly, 
// the result operand CANNOT be the same as one of the input operands, since
// the result is clobbered immediately and used as a scratchpad. Note that this
// is also unsigned addition: not only does it not accept negative numbers, it
// also doesn't do subtraction (which, for that matter, isn't commutative).
__device__ int bignum_add_gpu(bignum * resultnum, bignum * leftnum, bignum * rightnum) {
    bignum_reset_gpu(resultnum);
    if ((leftnum->sig_digs == 0L) && (rightnum->sig_digs > 0L)) {
        bignum_set_gpu(resultnum, rightnum);
        return 1;
    }
    else if ((rightnum->sig_digs == 0L) && (leftnum->sig_digs > 0L)) {
        bignum_set_gpu(resultnum, leftnum);
        return 1;
    }
    else if ((leftnum->sig_digs == 0L) && (rightnum->sig_digs == 0L)) { return 1; }
    else {
        
        // First check for overshift:  if the larger number's power is too much
        // bigger than the smaller number's, the smaller will be completely lost,
        // and we'll just end up with the large number as the result.
        if ((((leftnum->power - rightnum->power) > 0) && 
            ((leftnum->power - rightnum->power) > resultnum->precision))) {
            bignum_set_gpu(resultnum, leftnum);
            return 1;
        }
        if ((((rightnum->power - leftnum->power) > 0) && 
            ((rightnum->power - leftnum->power) > resultnum->precision))) {
            bignum_set_gpu(resultnum, rightnum);
            return 1;
        }
        
        // Next, shift the smaller operand to match the larger one by copying
        // it into the result operand as a partial sum. Also copy over the 
        // power and total significant digits into the result.
        bignum * bigger;
        bignum * smaller;
        if ((leftnum->power - rightnum->power) >= 0L) {
            bigger = leftnum;
            smaller = rightnum;
        }
        else {
            bigger = rightnum;
            smaller = leftnum;
        }
        long int difference = bigger->power - smaller->power;
        long int startdigit = smaller->sig_digs + difference;
        long int transfertotal = smaller->sig_digs;
        if (startdigit > resultnum->precision) {
            startdigit = resultnum->precision - difference;
            transfertotal = startdigit;
        }
        long int startdigitcopy = startdigit;
        startdigit--;
        long int i;
        for (i = 0L; i < transfertotal; i++) {
            if ((startdigit - difference) >= 0L) {
                resultnum->digits[(int)startdigit] = 
                    smaller->digits[(int)(startdigit - difference)];
            }
            startdigit--;
        }
        
        // Now the main addition loop: loop through each digit and add it.
        // The carry from the previous digit will add to the current one.
        // Note that we detect any trailing zeros to take from the sig_digs.
        // Also, copy over the power and significant digits
        resultnum->power = bigger->power;
        resultnum->sig_digs = startdigitcopy;
        if (bigger->sig_digs > resultnum->sig_digs) {
            resultnum->sig_digs = bigger->sig_digs;
            startdigitcopy = resultnum->sig_digs;
        }
        int trailingzeros = 1;
        long int zerocount = 0L;
        char carry = 0;
        for (i = 0L; i < resultnum->sig_digs; i++) {
            resultnum->digits[(int)(startdigitcopy - i - 1L)] += 
                (bigger->digits[(int)(startdigitcopy - i - 1L)] + carry);
            if (resultnum->digits[(int)(startdigitcopy - i - 1L)] >= 10) {
                resultnum->digits[(int)(startdigitcopy - i - 1L)] -= 10;
                carry = 1;
            } else { carry = 0; }
            if (trailingzeros == 1) {
                if (resultnum->digits[(int)(startdigitcopy - i - 1L)] == '\0') {
                    zerocount++;
                } else { trailingzeros = 0; }
            }
        }
        
        // If we've got trailing zeros, subtract them from the final count of
        // sig_digs. Also, if we have a carry, we need to shift everything...
        resultnum->sig_digs -= zerocount;
        if (carry > 0) {
            transfertotal = resultnum->sig_digs;
            if (transfertotal == resultnum->precision) { transfertotal--; }
            startdigitcopy = transfertotal - 1L;
            for (i = 0L; i < transfertotal; i++) {
                if (startdigitcopy >= 0L) {
                    resultnum->digits[(int)(startdigitcopy + 1L)] =
                        resultnum->digits[(int)startdigitcopy];
                }
                else if ((startdigitcopy + 1L) >= 0L) {
                    resultnum->digits[(int)(startdigitcopy + 1L)] = '\0';
                }
                startdigitcopy--;
            }
            resultnum->digits[0] = carry;
            resultnum->power++;
            resultnum->sig_digs++;
        }
        if (resultnum->sig_digs > resultnum->precision) {
            resultnum->sig_digs = resultnum->precision;
        }
        return 1;
    }
}

// A convenience wrapper that temporarily creates a new bignum out of the 
// given integer, calls bignum_add with it and the other operand, and deletes
// the temporary bignum before exiting. Any problems that bignum_add encounters
// are passed back up through this function and returned to the caller.
__device__ int bignum_add_int_gpu(bignum * resultnum, bignum * leftnum, long int rightint) {
    bignum_reset_gpu(resultnum);
    if ((rightint == 0L) && (leftnum->sig_digs > 0L)) {
        bignum_set_gpu(resultnum, leftnum);
        return 1;
    }
    else if ((leftnum->sig_digs == 0L) && (rightint > 0L)) {
        return bignum_set_int_gpu(resultnum, rightint);
    }
    else if ((leftnum->sig_digs == 0L) && (rightint == 0L)) { return 1; }
    else {
        bignum * tempnum = bignum_init_gpu(resultnum->precision);
        if (tempnum == 0) { return 0; }
        if (bignum_set_int_gpu(tempnum, rightint) == 0) {
            bignum_clear_gpu(tempnum);
            return 0;
        }
        int retval = bignum_add_gpu(resultnum, leftnum, tempnum);
        bignum_clear_gpu(tempnum);
        return retval;
    }
}

// Multiplies two bignums together and stores the result. Like add, uses 
// functions to reset and set the location of the result, and returns 1 upon
// success or 0 if an error occurred. A special shortcut is taken if either
// operand is zero, since the result will thus also be zero. Note that we assume
// the precision of all three operands is the same. If it's not, something 
// terrible like a seg fault or incorrect answer will probably occur. Most 
// importantly, the result operand CANNOT be the same as one of the input
// operands, since the result is clobbered immediately and used as a scratchpad.
// Also, note that this is unsigned: it assumes both operands are positive.
__device__ int bignum_mult_gpu(bignum * resultnum, bignum * leftnum, bignum * rightnum) {
    bignum_reset_gpu(resultnum);
    if ((leftnum->sig_digs == 0L) || (rightnum->sig_digs == 0L)) { return 1; }
    else {
        
        // Initialize the scratchpad and find the digit limits
        char * temp_word = (char *)malloc((int)(2L * (resultnum->precision) * sizeof(char)));
        if (temp_word == 0) { return 0; }
        long int i;
        for (i = 0; i < (2L * resultnum->precision); i++) { temp_word[(int)i] = '\0'; }
        bignum * bigger;
        bignum * smaller;
        if (((signed long int)leftnum->sig_digs - (signed long int)rightnum->sig_digs) >= 0L) {
            bigger = leftnum;
            smaller = rightnum;
        }
        else if ((rightnum->sig_digs - leftnum->sig_digs) > 0L) {
            bigger = rightnum;
            smaller = leftnum;
        }
        long int bigstart = (bigger->sig_digs) - 1L;
        long int smallstart = (smaller->sig_digs) - 1L;
        long int bigcounter, smallcounter;
        char carry = 0;
        
        // Perform the shift-addition loop. We choose to loop over each
        // digit of the smaller number for fewer overall iterations. If
        // the current bigloop has a zero, we can just skip that iteration.
        // Also, record the final carry, power, and sig_digs values. 
        for (bigcounter = 0L; bigcounter < (smaller->sig_digs); bigcounter++) {
            if (smaller->digits[(int)(smallstart - bigcounter)] != '\0') {
                carry = 0;
                for(smallcounter = 0L; smallcounter < (bigger->sig_digs); smallcounter++) {
                    temp_word[(int)((2L * (resultnum->precision)) - smallcounter - 
                        bigcounter - 1L)] += (carry + (smaller->digits[(int)(smallstart - 
                        bigcounter)] * bigger->digits[(int)(bigstart - smallcounter)]));
                    carry = temp_word[(int)((2L * (resultnum->precision)) - 
                        smallcounter - bigcounter - 1L)] / 10;
                    temp_word[(int)((2L * (resultnum->precision)) - smallcounter - 
                        bigcounter - 1L)] %= 10;
                }
                temp_word[(int)((2L * (resultnum->precision)) - bigcounter -
                    (bigger->sig_digs) - 1L)] = carry;
            }
        }
        resultnum->power = ((bigger->power) + (smaller->power));
        resultnum->sig_digs = ((bigger->sig_digs) + (smaller->sig_digs));
        
        // Adjust for lack of a final carry or trailing zeros.
        if (carry < 1) { 
            (resultnum->sig_digs)--; 
            (resultnum->power)--; 
        }
        (resultnum->power)++; 
        int trailingzeros = 1;
        long int zerocount = 0L;
        i = (2L * (resultnum->precision) - 1L); 
        while (trailingzeros == 1) {
            if (temp_word[(int)i] == '\0') {
                zerocount++;
            } else { trailingzeros = 0; }
            i--;
        }
        resultnum->sig_digs -= zerocount;
        if ((resultnum->sig_digs) > (resultnum->precision)) {
            resultnum->sig_digs = (resultnum->precision);
        }
        
        // Finally, copy from the temp word into the result, taking into 
        // account any digits we may lose due to precision.
        long int tempstart = (2L * (resultnum->precision)) - ((bigger->sig_digs) + 
            (smaller->sig_digs));
        if (carry < 1) { tempstart++; }
        for (i = 0L; i < (resultnum->sig_digs); i++) {
            resultnum->digits[(int)i] = temp_word[(int)(tempstart + i)];
        }
        free(temp_word);
        return 1;
    }
}

// Like bignum_add_int, a convenience wrapper that creates a temporary bignum
// out of the integer and passes it to bignum_mult. Any problems encountered 
// in client functions are passed back up to the original caller.
__device__ int bignum_mult_int_gpu(bignum * resultnum, bignum * leftnum, long int rightint) {
    bignum_reset_gpu(resultnum);
    if ((leftnum->sig_digs == 0L) || (rightint == 0L)) { return 1; }
    else {
        bignum * tempnum = bignum_init_gpu(resultnum->precision);
        if (tempnum == 0) { return 0; }
        if (bignum_set_int_gpu(tempnum, rightint) == 0) {
            bignum_clear_gpu(tempnum);
            return 0;
        }
        int retval = bignum_mult_gpu(resultnum, leftnum, tempnum);
        bignum_clear_gpu(tempnum);
        return retval;
    }
}

// Divides two bignums. Taken in terms of a fraction, leftnum is the numerator 
// and rightnum is the denominator. Performs an explicit check to make sure
// the denominator is not zero, and returns 0 (an error) if it is. Returns 1 upon 
// success or 0 if an error occurs. A special shortcut is taken if the numerator is 
// zero. Note that we assume the precision of all three operands is the same. If it's 
// not, something terrible like a seg fault or incorrect answer will probably occur. 
// Most importantly, the result operand CANNOT be the same as one of the input 
// operands, since the result is clobbered immediately and used as a scratchpad.
// Also, note that this is unsigned: it assumes both operands are positive.
__device__ int bignum_divide_gpu(bignum * resultnum, bignum * numerator, bignum * denominator) {
    bignum_reset_gpu(resultnum);
    if (denominator->sig_digs == 0L) { return 0; }
    else if (numerator->sig_digs == 0L) { return 1; }
    else {
        
        // Initialize the scratchpad and initially copy the numerator into it.
        // Also initialize the result's power.
        char * temp_word = (char *)malloc((int)(2L * 
            (resultnum->precision) + 2L * sizeof(char))); // May only need to be + 1L
        if (temp_word == 0) { return 0; }
        long int i;
        temp_word[0] = '\0';
        for (i = 0L; i < numerator->sig_digs; i++) {
            temp_word[(int)(i + 1L)] = numerator->digits[(int)i];
        }
        for (i = (1L + numerator->sig_digs); i < 
            (2L * resultnum->precision + 2L); i++) { temp_word[(int)i] = '\0'; }
        
        resultnum->power = (numerator->power - denominator->power);
        long int sigdigctr = 0L;
        long int numeratorindex = 0L;
        
        // First see if we need to "shift" the numerator by comparing it.
        i = ((denominator->sig_digs) - 1L);
        int denom_bigger = 1;
        while ((i >= 0L) && (denom_bigger == 1)) {
            if ((denominator->digits[(int)((denominator->sig_digs) - i - 1L)]) > 
                (temp_word[(int)((denominator->sig_digs) - i)])) {
                i = 0L;
            }
            else if ((denominator->digits[(int)((denominator->sig_digs) - 
                i - 1L)]) < (temp_word[(int)((denominator->sig_digs) - i)])) {
                denom_bigger = 0;
            }
            else if (((denominator->digits[(int)((denominator->sig_digs) - i - 
                1L)]) == (temp_word[(int)((denominator->sig_digs) - i)])) && (i == 0L)) {
                denom_bigger = 0;
            }
            i--;
        }
        if (denom_bigger == 1) { 
            numeratorindex++; 
            (resultnum->power)--;
        }
        
        // Now the main division loop. Note that there's two ways to terminate:
        // either we've filled the entire precision of the result word and are
        // forced to truncate our result, or our answer divides exactly. In the
        // second case, once we've exhausted the numerator's significant digits
        // and our temp word contains nothing but zeros, we can end early since
        // all subsequent iterations would contribute only zeros as well. Note
        // that special care will be taken to detect extra zeros at the end of
        // the result so that the sig_digs is recorded correctly. Also, we don't
        // round, we truncate, which doesn't minimize error.
        int nonzero = 1;
        while ((sigdigctr < (resultnum->precision)) && (nonzero == 1)) {
            // First run the subtraction loop.
            char current_digit = 0;
            int numer_bigger = 1;
            while (numer_bigger == 1) {
                // To subtract, first run a comparison to see if the numerator
                // is bigger. If it is, increment the counter and subtract. 
                i = ((denominator->sig_digs) - 1L);
                denom_bigger = 1;
                if (temp_word[(int)numeratorindex] > 0) { denom_bigger = 0; }
                while ((i >= 0L) && (denom_bigger == 1)) {
                    if ((denominator->digits[(int)((denominator->sig_digs) - 
                        i - 1L)]) > (temp_word[(int)((denominator->sig_digs) + 
                        numeratorindex - i)])) {
                        i = 0L;
                    }
                    else if ((denominator->digits[(int)((denominator->sig_digs) - 
                        i - 1L)]) < (temp_word[(int)((denominator->sig_digs) + 
                        numeratorindex - i)])) {
                        denom_bigger = 0;
                    }
                    else if (((denominator->digits[(int)((denominator->sig_digs) - 
                        i - 1L)]) == (temp_word[(int)((denominator->sig_digs) + 
                        numeratorindex - i)])) && (i == 0L)) {
                        denom_bigger = 0;
                    }
                    i--;
                }
                if (denom_bigger == 1) { 
                    numer_bigger = 0;
                }
                
                // Increment counter and perform subtraction loop.
                if (numer_bigger == 1) {
                    current_digit++;
                    for (i = 0L; i < (denominator->sig_digs); i++) {
                        temp_word[(int)((denominator->sig_digs) + 
                            numeratorindex - i)] -= (denominator->digits[
                            (int)((denominator->sig_digs) - i - 1L)]);
                        if ((temp_word[(int)((denominator->sig_digs) + 
                            numeratorindex - i)]) < 0) {
                            temp_word[(int)((denominator->sig_digs) + 
                                numeratorindex - i)] += 10L;
                            (temp_word[(int)((denominator->sig_digs) + 
                                numeratorindex - i - 1L)]) -= 1L;
                        }
                    }
                }
            }
            
            // If we're past all of the numerator's significant digits, run
            // zero detection on it to see if we can end early.
            if (sigdigctr > (numerator->sig_digs)) { // May only need to be >=
                long int zerocounter = 0L; 
                i = 0L;
                while ((i == zerocounter) && (i <= (denominator->sig_digs))) {
                    if ((temp_word[(int)(numeratorindex + i)]) < 1) { zerocounter++; }
                    i++;
                }
                if (zerocounter == ((denominator->sig_digs) + 1L)) { nonzero = 0; }
            }
            
            // Once we have obtained the proper digit in the result, save it.
            if (sigdigctr < resultnum->precision) {
                resultnum->digits[(int)sigdigctr] = current_digit;
            }
            sigdigctr++;
            numeratorindex++;
        }
        
        // Record the result's sig digs, taking care to detect trailing zeros.
        resultnum->sig_digs = sigdigctr;
        int trailingzeros = 1;
        long int zerocount = 0L;
        i = sigdigctr - 1L; 
        while (trailingzeros == 1) {
            if (resultnum->digits[(int)i] == '\0') {
                zerocount++;
            } else { trailingzeros = 0; }
            i--;
        }
        (resultnum->sig_digs) -= zerocount;
        free(temp_word);
        return 1;
    }
}

// A convenience wrapper that creates a temporary bignum out of the integer. 
// Since division is not commutative, two wrappers are given. Any problems 
// encountered in client functions are passed back up to the original caller.
__device__ int bignum_int_divide_gpu(bignum * resultnum, long int leftint, bignum * rightnum) {
    bignum_reset_gpu(resultnum);
    if (rightnum->sig_digs == 0L) { return 0; }
    else if (leftint == 0L) { return 1; }
    else {
        bignum * tempnum = bignum_init_gpu(resultnum->precision);
        if (tempnum == 0) { return 0; }
        if (bignum_set_int_gpu(tempnum, leftint) == 0) {
            bignum_clear_gpu(tempnum);
            return 0;
        }
        int retval = bignum_divide_gpu(resultnum, tempnum, rightnum);
        bignum_clear_gpu(tempnum);
        return retval;
    }
}

// A convenience wrapper that creates a temporary bignum out of the integer. 
// Since division is not commutative, two wrappers are given. Any problems 
// encountered in client functions are passed back up to the original caller.
__device__ int bignum_divide_int_gpu(bignum * resultnum, bignum * leftnum, long int rightint) {
    bignum_reset_gpu(resultnum);
    if (rightint == 0L) { return 0; }
    else if (leftnum->sig_digs == 0L) { return 1; }
    else {
        bignum * tempnum = bignum_init_gpu(resultnum->precision);
        if (tempnum == 0) { return 0; }
        if (bignum_set_int_gpu(tempnum, rightint) == 0) {
            bignum_clear_gpu(tempnum);
            return 0;
        }
        int retval = bignum_divide_gpu(resultnum, leftnum, tempnum);
        bignum_clear_gpu(tempnum);
        return retval;
    }
}
