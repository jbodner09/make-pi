import java.util.Arrays;

public class BigNum{
    public long power;
    public long sig_digs;
    public byte[] digits;
    public long precision;

    // Initialize a BigNum with the specified precision. We interpret 
    // having zero significant digits as the number having a value of zero.
    public BigNum(long precision) {
        this.digits = new byte[(int)precision];
        Arrays.fill(this.digits, (byte)0);
        this.precision = precision;
        this.power = 0L;
        this.sig_digs = 0L;
    }

    // Resets a BigNum's value to zero.
    public void Reset() {
        if (this.sig_digs > 0L) { Arrays.fill(this.digits, (byte)0); }
        this.power = 0L; 
        this.sig_digs = 0L;
    }

    // Set an instance of a bignum to an integer value. We assume that the number is 
    // non-negative since we only store unsigned numbers. Also, we handle zero specially 
    // by just resetting (again?) the result (shouldn't be re-using numbers...). Note that
    // we explicitly assume the number to convert fits within the max number of digits. If 
    // we try to convert a number bigger than we can store, it won't work.
    public void SetInt(long intval) {
        if (intval > 0L) {

            // Separate out the individual digits (stored backwards)
            byte[] temp_word = new byte[(int)this.precision];
            Arrays.fill(temp_word, (byte)0);
            long temp_int = intval;
            long counter = 0L;
            while (temp_int > 0L) {
                temp_word[(int)counter] = (byte)(temp_int % 10L);
                temp_int = temp_int / 10L;
                counter++;
            }

            // Detect any trailing zeros that we don't need to store
            this.power = counter - 1L;
            long leadingzeros = 0L;
            boolean hasleading = true;
            while (hasleading) {
                if (temp_word[(int)leadingzeros] != 0) { hasleading = false; }
                else { leadingzeros++; }
            }

            // Store final result into actual bignum variable
            for (temp_int = 0L; temp_int < (counter - leadingzeros); temp_int++) {
                this.digits[(int)temp_int] = temp_word[(int)(counter - temp_int - 1L)];
            }
            this.sig_digs = counter - leadingzeros;
        }
        else { this.Reset(); }
    }

    // Copy the value of another BigNum into this one. We don't assume
    // they're both the same precision; just use this precision.
    public void Copy(BigNum oldnum) {
        if ((oldnum.sig_digs) > 0L) {
            this.power = oldnum.power;
            this.sig_digs = ((oldnum.sig_digs > this.precision) ?
                (this.precision) : (oldnum.sig_digs));
            Arrays.fill(this.digits, (byte)0);
            this.digits = Arrays.copyOf(oldnum.digits, oldnum.digits.length);
        }
        else { this.Reset(); }
    }

    // Use StringBuilder to create the string representation of the number one digit 
    // at a time. There are a few cases:
    // power > significant digits: pad end with zeros
    // significant digits > power: fractional digit (non-integer)
    // power is negative: total value less than 1
    // The argument is the maximum number of significant digits to print.
    // If it's zero, then all available digits will be printed, maxing out at 
    // the precision of the number (the total amount is could possibly store).
    // Note that this is different from total digits printed: zeroes after a 
    // decimal point but before the first significant digit don't count, and we
    // make sure we print at least the integral part of the number (we only 
    // chop off fractional portions).
    public String Print(long maxdigits) {
        long i;
        long limit = this.sig_digs;
        StringBuilder resultstring = new StringBuilder();
        if (limit == 0L) { resultstring.append('0'); }
        else {
            if ((maxdigits > 0L) && (maxdigits < limit)) {
                limit = maxdigits;
            }
            if (this.power < 0L) {
                resultstring.append('0').append('.');
                for (i = 1L; i < (long)(-1L * (this.power)); i++) { resultstring.append('0'); }
                for (i = 0L; i < limit; i++) { resultstring.append((int)(this.digits[(int)i])); }
            }
            else if (this.sig_digs > (long)(this.power + 1L)) {
                for (i = 0L; i <= (long)(this.power); i++) { resultstring.append((int)(this.digits[(int)i])); }
                if (limit > (long)(this.power + 1L)) { resultstring.append('.'); }
                for (i = (long)(this.power + 1L); i < limit; i++) { resultstring.append((int)(this.digits[(int)i])); }
            }
            else {
                for (i = 0L; i < this.sig_digs; i++) { resultstring.append((int)(this.digits[(int)i])); }
            }
            if ((this.power > 0L) && ((long)(this.power + 1L) > this.sig_digs)) {
                for (i = 0L; i < ((this.power + 1L) - this.sig_digs); i++) { resultstring.append('0'); }
            }
        }
        return resultstring.toString();
    }

    // Adds two BigNums together and stores the result. Uses the functions to 
    // Reset and Copy into this number as the result. A special shortcut is
    // taken if either (or both) of the operands are zero. Note that it is possible 
    // for large additions to cause underflow to zero. In that case, special care is
    // taken to make sure the proper input operand is used. Note that we assume the
    // precision of all three operands is the same. If it's not, something terrible
    // like a seg fault or incorrect answer will probably occur. Most importantly, 
    // the result operand CANNOT be the same as one of the input operands, since
    // the result is clobbered immediately and used as a scratchpad. Note that this
    // is also unsigned addition: not only does it not accept negative numbers, it
    // also doesn't do subtraction (which, for that matter, isn't commutative).
    public void Add(BigNum leftnum, BigNum rightnum) {
        this.Reset();
        if ((leftnum.sig_digs == 0L) && (rightnum.sig_digs > 0L)) { this.Copy(rightnum); }
        else if ((rightnum.sig_digs == 0L) && (leftnum.sig_digs > 0L)) { this.Copy(leftnum); }
        else if ((leftnum.sig_digs == 0L) && (rightnum.sig_digs == 0L)) { this.Reset(); }
        else {

            // First check for overshift:  if the larger number's power is too much
            // bigger than the smaller number's, the smaller will be completely lost,
            // and we'll just end up with the large number as the result.
            if (((leftnum.power - rightnum.power) > 0L) &&
                ((leftnum.power - rightnum.power) > this.precision)) {
                this.Copy(leftnum);
                return;
            }
            if (((rightnum.power - leftnum.power) > 0L) &&
                ((rightnum.power - leftnum.power) > this.precision)) {
                this.Copy(rightnum);
                return;
            }

            // Next, shift the smaller operand to match the larger one by copying
            // it into the result operand as a partial sum. Also copy over the 
            // power and total significant digits into the result.
            BigNum bigger = new BigNum(this.precision);
            BigNum smaller = new BigNum(this.precision);
            if ((leftnum.power - rightnum.power) >= 0L) {
                bigger.Copy(leftnum);
                smaller.Copy(rightnum);
            }
            else {
                bigger.Copy(rightnum);
                smaller.Copy(leftnum);
            }
            long difference = bigger.power - smaller.power;
            long startdigit = smaller.sig_digs + difference;
            long transfertotal = smaller.sig_digs;
            if (startdigit > this.precision) {
                startdigit = this.precision - difference;
                transfertotal = startdigit;
            }
            long startdigitcopy = startdigit;
            startdigit--;
            long i;
            for (i = 0L; i < transfertotal; i++) {
                if ((startdigit - difference) >= 0L) {
                    this.digits[(int)startdigit] = smaller.digits[(int)(startdigit - difference)];
                }
                startdigit--;
            }

            // Now the main addition loop: loop through each digit and add it.
            // The carry from the previous digit will add to the current one.
            // Note that we detect any trailing zeros to take from the sig_digs.
            // Also, copy over the power and significant digits
            this.power = bigger.power;
            this.sig_digs = startdigitcopy;
            if (bigger.sig_digs > this.sig_digs) {
                this.sig_digs = bigger.sig_digs;
                startdigitcopy = this.sig_digs;
            }
            boolean trailingzeros = true;
            long zerocount = 0L;
            byte carry = 0;
            for (i = 0L; i < this.sig_digs; i++) {
                this.digits[(int)(startdigitcopy - i - 1L)] += 
                    (byte)(bigger.digits[(int)(startdigitcopy - i - 1L)] + carry);
                if (this.digits[(int)(startdigitcopy - i - 1L)] >= 10) {
                    this.digits[(int)(startdigitcopy - i - 1L)] -= 10;
                    carry = 1;
                }
                else { carry = 0; }
                if (trailingzeros) {
                    if (this.digits[(int)(startdigitcopy - i - 1L)] == 0) {
                        zerocount++;
                    }
                    else { trailingzeros = false; }
                }
            }

            // If we've got trailing zeros, subtract them from the final count of
            // sig_digs. Also, if we have a carry, we need to shift everything...
            this.sig_digs -= zerocount;
            if (carry > 0) {
                transfertotal = this.sig_digs;
                if (transfertotal == this.precision) { transfertotal--; }
                startdigitcopy = transfertotal - 1L;
                for (i = 0L; i < transfertotal; i++) {
                    if (startdigitcopy >= 0L) {
                        this.digits[(int)(startdigitcopy + 1L)] = this.digits[(int)startdigitcopy];
                    }
                    else if ((startdigitcopy + 1L) >= 0L) {
                        this.digits[(int)(startdigitcopy + 1L)] = 0;
                    }
                    startdigitcopy--;
                }
                this.digits[0] = carry;
                this.power++;
                this.sig_digs++;
            }
        }
    }

    // A convenience wrapper that temporarily creates a new BigNum out of the 
    // given integer then calls Add with it and the other operand.
    public void AddInt(BigNum leftnum, long rightint) {
        this.Reset();
        if ((rightint == 0L) && (leftnum.sig_digs > 0L)) { this.Copy(leftnum); }
        else if ((leftnum.sig_digs == 0L) && (rightint > 0L)) { this.SetInt(rightint); }
        else if ((leftnum.sig_digs == 0L) && (rightint == 0L)) { return; }
        else {
            BigNum tempnum = new BigNum(this.precision);
            tempnum.SetInt(rightint);
            this.Add(leftnum, tempnum);
        }
    }

    // Multiplies two bignums together and stores the result. Like add, uses 
    // functions to reset and set the location of the result. A special shortcut is 
    // taken if either operand is zero, since the result will thus also be zero. 
    // Note that we assume  the precision of all three operands is the same. If it's
    // not, something terrible like a seg fault or incorrect answer will probably occur. 
    // Most importantly, the result operand CANNOT be the same as one of the input
    // operands, since the result is clobbered immediately and used as a scratchpad.
    // Also, note that this is unsigned: it assumes both operands are positive.
    public void Mult(BigNum leftnum, BigNum rightnum) {
        this.Reset();
        if ((leftnum.sig_digs == 0L) || (rightnum.sig_digs == 0L)) { return; }
        else {

            // Initialize the scratchpad and find the digit limits
            byte[] temp_word = new byte[(int)(2L * this.precision)];
            BigNum bigger = new BigNum(this.precision);
            BigNum smaller = new BigNum(this.precision);
            if ((leftnum.sig_digs - rightnum.sig_digs) >= 0L) {
                bigger.Copy(leftnum);
                smaller.Copy(rightnum);
            }
            else {
                bigger.Copy(rightnum);
                smaller.Copy(leftnum);
            }
            long bigstart = bigger.sig_digs - 1L;
            long smallstart = smaller.sig_digs - 1L;
            long bigcounter, smallcounter;
            byte carry = 0;

            // Perform the shift-addition loop. We choose to loop over each
            // digit of the smaller number for fewer overall iterations. If
            // the current bigloop has a zero, we can just skip that iteration.
            // Also, record the final carry, power, and sig_digs values. 
            for (bigcounter = 0L; bigcounter < (smaller.sig_digs); bigcounter++) {
                if (smaller.digits[(int)(smallstart - bigcounter)] != 0) {
                    carry = 0;
                    for (smallcounter = 0L; smallcounter < (bigger.sig_digs); smallcounter++) {
                        temp_word[(int)((2L * (this.precision)) - smallcounter - bigcounter - 1L)] += 
                            (byte)(carry + (smaller.digits[(int)(smallstart - bigcounter)] * 
                            bigger.digits[(int)(bigstart - smallcounter)]));
                        carry = (byte)(temp_word[(int)((2L * (this.precision)) - smallcounter - bigcounter - 1L)] / 10);
                        temp_word[(int)((2L * (this.precision)) - smallcounter - bigcounter - 1L)] %= 10;
                    }
                    temp_word[(int)((2L * (this.precision)) - bigcounter - (bigger.sig_digs) - 1L)] = carry;
                }
            }
            this.power = ((bigger.power) + (smaller.power));
            this.sig_digs = ((bigger.sig_digs) + (smaller.sig_digs));

            // Adjust for lack of a final carry or trailing zeros.
            if (carry < 1) {
                this.sig_digs--;
                this.power--;
            }
            this.power++;
            boolean trailingzeros = true;
            long zerocount = 0L;
            long i = 2L * this.precision - 1L;
            while (trailingzeros) {
                if (temp_word[(int)i] == 0) {
                    zerocount++;
                }
                else { trailingzeros = false; }
                i--;
            }
            this.sig_digs -= zerocount;
            if (this.sig_digs > this.precision) {
                this.sig_digs = this.precision;
            }

            // Finally, copy from the temp word into the result, taking into 
            // account any digits we may lose due to precision. Note that we
            // can't use the CopyTo function since it only works with entire 
            // arrays, and we're only copying up to half of the temp one.
            long tempstart = (2L * (this.precision)) - (bigger.sig_digs + smaller.sig_digs);
            if (carry < 1) { tempstart++; }
            long j = 0L;
            for (j = 0L; j < (this.sig_digs); j++) {
                this.digits[(int)j] = temp_word[(int)(tempstart + j)];
            }
        }
    }

    // Like AddInt, a convenience wrapper that creates a temporary BigNum
    // out of the integer and passes it to Mult. 
    public void MultInt(BigNum leftnum, long rightint) {
        this.Reset();
        if ((leftnum.sig_digs == 0L) || (rightint == 0L)) { return; }
        else {
            BigNum tempnum = new BigNum(this.precision);
            tempnum.SetInt(rightint);
            this.Mult(leftnum, tempnum);
        }
    }

    // Divides two BigNums. Taken in terms of a fraction, leftnum is the numerator 
    // and rightnum is the denominator. Performs an explicit check to make sure
    // the denominator is not zero, and returns 0 if it is. A special shortcut is taken 
    // if the numerator is zero. Note that we assume the precision of all three operands
    // is the same. If it's not, something terrible like a seg fault or incorrect answer 
    // will probably occur. Most importantly, the result operand CANNOT be the same as 
    // one of the input operands, since the result is clobbered immediately and used as 
    // a scratchpad. Also, note that this is unsigned: it assumes both operands are positive.
    public void Divide(BigNum numerator, BigNum denominator) {
        this.Reset();
        if ((numerator.sig_digs == 0L) || (denominator.sig_digs == 0L)) { return; }
        else {

            // Initialize the scratchpad and initially copy the numerator into it.
            // Also initialize the result's power. Can't use CopyTo
            int[] temp_word = new int[(int)(2L * (this.precision) + 2L)]; // May only need to be +1L
            long j;
            for (j = 0L; j < numerator.sig_digs; j++) {
                temp_word[(int)(j + 1L)] = (int)numerator.digits[(int)j];
            }
            this.power = (numerator.power - denominator.power);
            long sigdigctr = 0L;
            long numeratorindex = 0L;

            // First see if we need to "shift" the numerator by comparing it.
            long i = ((denominator.sig_digs) - 1L);
            boolean denom_bigger = true;
            while ((i >= 0L) && (denom_bigger)) {
                if ((denominator.digits[(int)((denominator.sig_digs) - i - 1L)]) >
                    (byte)(temp_word[(int)((denominator.sig_digs) - i)])) { i = 0L; }
                else if ((denominator.digits[(int)((denominator.sig_digs) - i - 1L)]) < 
                    (byte)(temp_word[(int)((denominator.sig_digs) - i)])) { denom_bigger = false; }
                else if (((denominator.digits[(int)((denominator.sig_digs) - i - 1L)]) == 
                    (byte)(temp_word[(int)((denominator.sig_digs) - i)])) && (i == 0L)) { denom_bigger = false; }
                i--;
            }
            if (denom_bigger) {
                numeratorindex++;
                this.power--;
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
            boolean nonzero = true;
            while ((sigdigctr < (this.precision)) && (nonzero)) {
                // First run the subtraction loop.
                byte current_digit = 0;
                boolean numer_bigger = true;
                while (numer_bigger) {

                    // To subtract, first run a comparison to see if the numerator
                    // is bigger. If it is, increment the counter and subtract. 
                    i = ((denominator.sig_digs) - 1L);
                    denom_bigger = true;
                    if (temp_word[(int)numeratorindex] > 0) { denom_bigger = false; }
                    while ((i >= 0L) && (denom_bigger)) {
                        if ((denominator.digits[(int)((denominator.sig_digs) - i - 1L)]) > 
                            (byte)(temp_word[(int)((denominator.sig_digs) + numeratorindex - i)])) { i = 0L; }
                        else if ((denominator.digits[(int)((denominator.sig_digs) - i - 1L)]) < 
                            (byte)(temp_word[(int)((denominator.sig_digs) + numeratorindex - i)])) { denom_bigger = false; }
                        else if (((denominator.digits[(int)((denominator.sig_digs) - i - 1L)]) == 
                            (byte)(temp_word[(int)((denominator.sig_digs) + numeratorindex - i)])) && (i == 0L)) { denom_bigger = false; }
                        i--;
                    }
                    if (denom_bigger) { numer_bigger = false; }

                    // Increment counter and perform subtraction loop.
                    if (numer_bigger) {
                        current_digit++;
                        for (j = 0L; j < (denominator.sig_digs); j++) {
                            temp_word[(int)((denominator.sig_digs) + numeratorindex - j)] -= 
                                (int)(denominator.digits[(int)((denominator.sig_digs) - j - 1L)]);
                            if ((temp_word[(int)((denominator.sig_digs) + numeratorindex - j)]) < 0) {
                                temp_word[(int)((denominator.sig_digs) + numeratorindex - j)] += 10;
                                temp_word[(int)((denominator.sig_digs) + numeratorindex - j - 1L)] -= 1;
                            }
                        }
                    }
                }

                // If we're past all of the numerator's significant digits, run
                // zero detection on it to see if we can end early.
                if (sigdigctr > (numerator.sig_digs)) { // May only need to be >=
                    long zerocounter = 0L;
                    j = 0L;
                    while ((j == zerocounter) && (j <= (denominator.sig_digs))) {
                        if ((temp_word[(int)(numeratorindex + j)]) < 1) { zerocounter++; }
                        j++;
                    }
                    if (zerocounter == ((denominator.sig_digs) + 1L)) { nonzero = false; }
                }

                // Once we have obtained the proper digit in the result, save it.
                if (sigdigctr < this.precision) {
                    this.digits[(int)sigdigctr] = current_digit;
                }
                sigdigctr++;
                numeratorindex++;
            }

            // Record the result's sig digs, taking care to detect trailing zeros.
            this.sig_digs = sigdigctr;
            boolean trailingzeros = true;
            long zerocount = 0L;
            i = sigdigctr - 1L;
            while (trailingzeros) {
                if (this.digits[(int)i] == 0) { zerocount++; }
                else { trailingzeros = false; }
                i--;
            }
            this.sig_digs -= zerocount;
        }
    }

    // A convenience wrapper that creates a temporary BigNum out of the integer. 
    // Since division is not commutative, two wrappers are given. 
    public void IntDivide(long leftint, BigNum rightnum) {
        this.Reset();
        if ((leftint == 0L) || (rightnum.sig_digs == 0L)) { return; }
        else {
            BigNum tempnum = new BigNum(this.precision);
            tempnum.SetInt(leftint);
            this.Divide(tempnum, rightnum);
        }
    }

    // A convenience wrapper that creates a temporary BigNum out of the integer. 
    // Since division is not commutative, two wrappers are given. 
    public void DivideInt(BigNum leftnum, long rightint) {
        this.Reset();
        if ((leftnum.sig_digs == 0L) || (rightint == 0L)) { return; }
        else {
            BigNum tempnum = new BigNum(this.precision);
            tempnum.SetInt(rightint);
            this.Divide(leftnum, tempnum);
        }
    }
}
