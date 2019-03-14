#Version 1: Avoiding Pointers, in `make_pi.c`

One of the most interesting quirks of this code is that I allocate all the space needed for all the scratch variables up front.  That means that the amount of RAM your computer has is directly proportional to the precision of your calculation!  (Well, until we hit the double limit mentioned on the main page).  I guess if you've ever wanted an easy way to see what happens to everything else on your computer when you request 100% of your available RAM, this is how you'll do it!  

##Version 1.1: Avoiding compilation too, in `pi_calculate.m`

Also, in case you don't have access to a C compiler, here's the Matlab code that not only does the Simpson's Rule method, but also does the infinite series method, so you can compare how many iterations of each you need to get to get comparable, accurate results.  Save it as an m file and run it.  (I'll give you a hint:  The Simpson's version still peaks at 20000, but no matter how high you make the series version go, your computer will run out of memory before you can make it even slightly comparable, which is why the Simpson's rule is superior for those of us with limited computing resources).  
