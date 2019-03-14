public class Calculation extends Thread {
    public long threadid;
    public long lowlimit;
    public long highlimit;
    public long totaliterations;
    public long max_digits;
    public BigNum finaltrap;
    public BigNum finalmid;

    public Calculation(long max_digits)
    {
        this.threadid = 0L;
        this.lowlimit = 0L;
        this.highlimit = 0L;
        this.totaliterations = 0L;
        this.max_digits = max_digits;
        this.finaltrap = new BigNum(this.max_digits);
        this.finalmid = new BigNum(this.max_digits);
    }

    public void run() {

        // Initialize needed variables
        BigNum trap = new BigNum(this.max_digits);
        BigNum mid = new BigNum(this.max_digits);
        BigNum inverseiterations = new BigNum(this.max_digits);
        BigNum temp_holder = new BigNum(this.max_digits);
        BigNum temp_holder2 = new BigNum(this.max_digits);
        BigNum inc = new BigNum(this.max_digits);
        BigNum leftrect = new BigNum(this.max_digits);
        BigNum rightrect = new BigNum(this.max_digits);

        // Initialize values of needed variables
        temp_holder.SetInt(this.totaliterations);
        inverseiterations.IntDivide(1L, temp_holder);
        temp_holder.Reset();
        long i;
        long k = this.lowlimit;
        temp_holder.DivideInt(inverseiterations, 2L);
        inc.SetInt(k);
        temp_holder2.Mult(inc, inverseiterations);
        inc.Reset();
        inc.Copy(temp_holder2);
        temp_holder2.Reset();
        temp_holder2.Add(inc, temp_holder);
        inc.Reset();
        inc.Copy(temp_holder2);
        temp_holder2.Reset();
        temp_holder.Reset();

        // Main iteration loop. Note that the values of inverseiterations, inc, 
        // mid, and trap are preserved across loop iterations, as is counter k.
        // inverseiterations is a constant that is stored for simplicity. Man, 
        // this is looking more and more like assembly...
        for (i = this.lowlimit; i < this.highlimit; i++)
        {

            // First, the trapezoid rule is used to estimate pi
            leftrect.Reset();
            leftrect.SetInt(k);
            temp_holder2.Mult(leftrect, inverseiterations);
            leftrect.Reset();
            leftrect.Copy(temp_holder2);
            temp_holder2.Reset();
            k++;
            rightrect.Reset();
            rightrect.SetInt(k);
            temp_holder2.Mult(rightrect, inverseiterations);
            rightrect.Reset();
            rightrect.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder.Add(leftrect, rightrect);
            temp_holder2.DivideInt(temp_holder, 2L);
            temp_holder.Reset();
            temp_holder.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder2.Mult(temp_holder, temp_holder);
            temp_holder.Reset();
            temp_holder.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder2.AddInt(temp_holder, 1L);
            temp_holder.Reset();
            temp_holder.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder2.IntDivide(1L, temp_holder);
            temp_holder.Reset();
            temp_holder.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder2.Mult(temp_holder, inverseiterations);
            temp_holder.Reset();
            temp_holder.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder2.Add(trap, temp_holder);
            trap.Reset();
            trap.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder.Reset();

            // Next, the midpoint rule is also used to estimate pi
            temp_holder.Copy(inc);
            temp_holder2.Add(inc, inverseiterations);
            inc.Reset();
            inc.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder2.Mult(temp_holder, temp_holder);
            temp_holder.Reset();
            temp_holder.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder2.AddInt(temp_holder, 1L);
            temp_holder.Reset();
            temp_holder.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder2.IntDivide(1L, temp_holder);
            temp_holder.Reset();
            temp_holder.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder2.Mult(temp_holder, inverseiterations);
            temp_holder.Reset();
            temp_holder.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder2.Add(mid, temp_holder);
            mid.Reset();
            mid.Copy(temp_holder2);
            temp_holder2.Reset();
            temp_holder.Reset();
        }

        // Save partial result and exit
        this.finaltrap.Copy(trap);
        this.finalmid.Copy(mid);
    }
}
