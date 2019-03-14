/*Since Java can be compiled from the command line, you don't need an IDE! 
To compile, you'll first need to do javac Calculation.java
If javac isn't in the environment, you'll need to navigate your terminal
to the javac folder (usually something like Java/jdk1.xxxxx/bin) and run
it from there, fully qualifying the file name of the source file.
Then, do javac Program.java
If you have to run it from the javac folder, it probably won't find the
class created for Calculation, since that directory isn't on the classpath.
In that case, you can temporarily add it by typing
javac -cp path_to_location_of_source_and_classes Program.java
again fully qualifying the path to Program.java as necessary. 
Then to run it, just type java Program 20000 8
which can be done from the folder that the class files are located in
since java should be in the environment variable. */
import java.lang.Exception;

public final class Program {
    public static void main(String[] args) {
        
        // Obtain command line arguments
        long iterations = 20000L;
        try
        {
            iterations = Long.parseLong(args[0]);
        }
        catch (Exception e)
        {
            iterations = 20000L;
        }
        int num_threads = 8;
        try
        {
            num_threads = Integer.parseInt(args[1]);
        }
        catch (Exception e)
        {
            num_threads = 8;
        }

        // Initialize global storage
        int i;
        Calculation[] threadslist = new Calculation[num_threads];

        // Split off worker threads. When dividing the work, if the number of 
        // threads does not evenly divide into the desired number of iterations,
        // give any extra iterations to the final thread. This gives the final
        // thread at most (num_threads - 1) extra iterations. 
        long startclock = System.nanoTime();
        for (i = 0; i < num_threads; i++)
        {
            Calculation current_thread = new Calculation();
            current_thread.threadid = (long)i;
            current_thread.lowlimit = (long)i * (iterations / (long)num_threads);
            current_thread.highlimit = (((i + 1) == num_threads) ? iterations :
                ((long)(i + 1) * (iterations / (long)num_threads)));
            current_thread.totaliterations = iterations;
            threadslist[i] = current_thread;
            try
            {
                current_thread.start();
            }
            catch (Exception e)
            {
                System.out.println("Error creating thread. Now terminating.");
                System.exit(-2);
            }
        }

        // Wait for all the threads to return, and after each of the 
        // worker threads end, clean up each of the partial sums
        double mid = 0.0;
        double trap = 0.0;
        for (i = 0; i < num_threads; i++)
        {
            try 
            {
                threadslist[i].join();
            } 
            catch (Exception e) 
            {
                System.out.println("Error waiting for thread to end. Now terminating.");
                System.exit(-3);
            }
            trap = trap + threadslist[i].finaltrap;
            mid = mid + threadslist[i].finalmid;
        }
        long endclock = System.nanoTime();

        // Finally, Simpson's Rule is applied
        double simp = (((2.0 * mid) + trap) / 3.0) * 4.0;
        System.out.format("The calculated value of pi is %.21f%n", simp);
        System.out.println("The actual value of pi is     3.141592653589793238463");
        System.out.format("The time taken to calculate this was %.2f seconds%n",
            ((double)(endclock - startclock) / 1000000000.0));
    }
}
