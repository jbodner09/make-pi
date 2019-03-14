using System;
using System.Threading;
using System.Diagnostics;

namespace WinPiSharp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Obtain command line arguments
            long iterations = 20000L;
            try
            {
                iterations = Convert.ToInt64(args[0]);
            }
            catch (Exception)
            {
                iterations = 20000L;
            }
            int num_threads = 8;
            try
            {
                num_threads = Convert.ToInt32(args[1]);
            }
            catch (Exception)
            {
                num_threads = 8;
            }

            // Initialize global storage
            int i;
            Calculation[] argslist = new Calculation[num_threads];
            Thread[] threadslist = new Thread[num_threads];

            // Split off worker threads. When dividing the work, if the number of 
            // threads does not evenly divide into the desired number of iterations,
            // give any extra iterations to the final thread. This gives the final
            // thread at most (num_threads - 1) extra iterations. 
            Stopwatch clock = Stopwatch.StartNew();
            for (i = 0; i < num_threads; i++)
            {
                Calculation current_args = new Calculation();
                current_args.threadid = (long)i;
                current_args.lowlimit = (long)i * (iterations / (long)num_threads);
                current_args.highlimit = (((i + 1) == num_threads) ? iterations :
                    ((long)(i + 1) * (iterations / (long)num_threads)));
                current_args.totaliterations = iterations;
                argslist[i] = current_args;
                Thread current_thread = new Thread(current_args.DoCalculate);
                threadslist[i] = current_thread;
                try
                {
                    current_thread.Start();
                }
                catch (Exception)
                {
                    Console.WriteLine("Error creating thread. Now terminating.");
                    Environment.Exit(-2);
                }
            }

            // Wait for all the threads to return, and after each of the 
            // worker threads end, clean up each of the partial sums
            double mid = 0.0;
            double trap = 0.0;
            for (i = 0; i < num_threads; i++)
            {
                threadslist[i].Join();
                trap = trap + argslist[i].finaltrap;
                mid = mid + argslist[i].finalmid;
            }
            clock.Stop();

            // Finally, Simpson's Rule is applied
            double simp = (((2.0 * mid) + trap) / 3.0) * 4.0;
            Console.WriteLine("The calculated value of pi is {0:F21}", simp);
            Console.WriteLine("The actual value of pi is     3.141592653589793238463");
            Console.WriteLine("The time taken to calculate this was {0:F2} seconds\n",
                (clock.ElapsedMilliseconds / 1000.0));
        }

        public class Calculation
        {
            public long threadid;
            public long lowlimit;
            public long highlimit;
            public long totaliterations;
            public double finaltrap;
            public double finalmid;

            public Calculation()
            {
                threadid = 0L;
                lowlimit = 0L;
                highlimit = 0L;
                totaliterations = 0L;
                finaltrap = 0.0;
                finalmid = 0.0;
            }

            public void DoCalculate() {

                // Initialize needed variables
                double trap = 0.0, mid = 0.0;
                double inverseiterations = 1.0 / (double)this.totaliterations;
                long i = 0L, k = this.lowlimit;
                double inc = (inverseiterations / 2.0) + inverseiterations * (double)k;

                for (i = this.lowlimit; i < this.highlimit; i++) {
                    // First, the trapezoid rule is used to estimate pi
                    double leftrect = (double)k * inverseiterations;
                    k++;
                    double rightrect = (double)k * inverseiterations;
                    double trapsquared = ((leftrect + rightrect) / 2.0) * ((leftrect + rightrect) / 2.0);
                    double traptemp = (1.0 / (1.0 + trapsquared)) * inverseiterations;
                    trap = trap + traptemp;

                    // Next, the midpoint rule is also used to estimate pi
                    double inctemp = inc;
                    inc = inc + inverseiterations;
                    double midtemp = (1.0 / (1.0 + (inctemp * inctemp))) * inverseiterations;
                    mid = mid + midtemp;
                }

                // Save partial result and exit
                this.finaltrap = trap;
                this.finalmid = mid;
            }
        }
    }
}
