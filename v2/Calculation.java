public class Calculation extends Thread {
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

    public void run() {

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
