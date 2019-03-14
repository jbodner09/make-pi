# make-pi

The code presented here is a journey through my education, in honor of the 10th anniversary of me learning how to code (and 10 years of cramming 10,000 words into Facebook posts that are too long for anybody to actually read).  It's presented in all its horrifically architected, bug-filled glory, and shouldn't ever be downloaded or compiled.  I mean, it should actually compile, but really, please don't.  

## The Pi Manifesto

In honor of Pi Day (and boredom and not wanting to study for finals), we're going to learn how to calculate pi without even knowing how to do calculus (really, we're going to USE calculus, but you don't have to know what it means because I'm going to break it down that much for you.  I'm just that good.)  I've always wanted to do this, and I finally know enough calculus to actually understand why this works, so I'm going to share my knowledge with you.  Thanks to http://mb-soft.com/public3/pi.html for getting me started with this endeavor, and MATLAB for helping me prove it.  

To start with, we know that pi times the diameter of a circle equals its circumference.  We're going to look at a special circle called the Unit Circle.  The unit circle has a radius of 1, which means that its circumference equals 2 times pi.  We also know that a circle is the same as a 360 degree angle, which means that 180 degrees, or half the circle, is equal to pi.  

In order to understand a little more about where this comes from, we're going to introduce a unit of measure called the radian.  In a circle, one radian is defined as the angle through which the length of the intersected arc is equal to the radius of the circle.  This happens at around 57.3 degrees.  To put it another way, on the unit circle, an angle of 57.3 degrees, or 1 radian, would give us an arc length of 1.  

How does this relate to pi?  Well, if 1 radian equals 57.3 degrees, then 1.57 radians equals 90 degrees, which is a right angle and one quarter of a circle.  1.57 also happens to be pi divided by 2.  Go ahead, check it in a calculator.  This means that 360 degrees is the same as 2 times pi radians, as we said above.  

For the rest of this explanation, we're going to look at 45 degrees, or pi / 4 radians.  Why 45 degrees?  Well, the tangent of 45 degrees is 1 according to right triangle trigonometry.  The tangent of an angle is a ratio of the side of the triangle opposite of the angle to the side of the triangle adjacent to the angle.  At 45 degrees, the lengths of both of those sides are the same, so the ratio is just 1:1, which makes the tangent of 45 degrees (or pi / 4 radians) equal to 1.  Going further, we can say that the *inverse tangent*, or arctangent, of *that ratio* is equal to pi / 4.  

Now we're getting somewhere.  We know that arctan (1) = pi / 4, but now we want to know WHY that's true.  Let's call arctan (x) a function f(x).  We know that any function can be graphed.  We also know that a function has a certain slope at any given point on the curve.  Well, let's say we graph the value of that slope.  That graph is called the *derivative* of the function, which we'll call f prime(x).  To explain that a little further, if the *slope* of f(x) is positive, then the *value* of f prime(x) is positive.  It's a little tricky to explain that without pictures, so just take my word for it.  

In order to get from f prime(x) back to f(x), we take what's called the *integral* of the function.  The integral is essentially the area underneath a graph between two points.  This property is best demonstrated with the Fundamental Theorem of Calculus.  Yep, it's fundamental, which means it's important.  It says that "the integral from points a to b (b being larger than a) of f prime(x) = f(b) – f(a)."  In other words, the integral at two points is the same as subtracting the *antiderivative* at one of those points from the antiderivative at the other point.  Yeah, I know I've lost a few of you there, but it's actually pretty simple.  

We called arctan(x) our f(x).  Well, the area underneath the graph from x = 0 to x = 1 of the derivative of f(x) = pi / 4.  That is the same as saying arctan(1) – arctan(0) = pi / 4 because arctan(0) = 0.  "What's with all this area under the graph business?" most of you are probably wondering.  Well, that's really how we calculate pi.  That's where it all comes from.  Taking the integral (area under the graph from 0 to 1) of the derivative of arctan(x) is where that pi / 4 comes from to begin with.  

Now comes the actual calculating part.  The derivative of arctan(x) = 1 / (1 + x squared).  Just take my word for it.  So, the integral from 0 to 1 of 1 / (1 + x squared) = arctan(1) – arctan(0) by the Fundamental theorem of calculus, which is basically what I've been trying to explain in the last three paragraphs.  So now we've got a function 1 / (1 + x squared) and we want to find the area underneath the graph of that function from 0 to 1 on the x-axis.  

Actually, I think I'll integrate 1 / (1 + x squared) now just to prove to you that its antiderivative really is arctan(x).  Don't worry if you don't understand what I do here.  We'll start with a trig substitution.  On the bottom, we've got 1 squared plus x squared, so for x, we can substitute one times tan(theta), or just tan(theta).  When we take an integral, we also have to multiply by the derivative of that substitution, and the derivative of tan(theta) is secant squared (theta), or sec^2(theta).  So now, our fraction looks like sec^2(theta) / (1 + tan^2(theta)).  There is a trig identity which states that sec^2(x) = 1 + tan^2(x), so now our fraction is sec^2(theta) / sec^2(theta), which is just 1.  The integral of 1 is just theta.  Remember when we said that x = tan(theta)?  Remember when we said that x was just a ratio of those sides, and that arctan(x) = the angle?  Well, theta is just the angle, so theta = arctan(x), which is how we got arctan(x) to be the antiderivative of 1 / (1 + x squared).  That was the hard part.  

Now we're back to the area under 1 / (1 + x^2).  We can't find the exact area, because the graph of 1 / (1 + x^2) is a curve (the graph also approaches infinity as x approaches 0, but that's not really important here).  Our main tactic for finding the area here is going to really be an estimation.  We are going to break up that area into a certain number of rectangles.  We know how to find the area of a rectangle (width times height).  The height of the rectangle is the value of the graph at a certain point, and the width is just 1 / the number of rectangles we make (It's 1 over it because the graph goes from 0 to 1, remember?)  Let's say we break up the area into 4 rectangles.  The width of each rectangle is going to be 1/4.  That means we've made five points at 0, 1/4, 1/2, 3/4, and 1.  

We're actually going to use two different methods of finding the height.  The first method is more accurate, and it's called the midpoint rule.  Remember when we said the height of the rectangle is just the value of the function at a point?  Well, that point is going to be half way between each of the five points mentioned above.  That means to find our four heights, we're going to plug 1/8, 3/8, 5/8, and 7/8 into 1 / (1 + x^2).  Then we multiply those four resulting values by our width (in this case 1/4), and add those four resulting values up.  Know what you've just found?  You've just estimated the area under the graph, which means you've just estimated the integral of the function, which means you've just estimated the value of arctan(1).  You can plug arctan(1) into a calculator to compare; it actually should be within about .15 of the actual value.  Don't forget that arctan(1) is pi / 4, so multiply your final answer by 4 to get your estimation of pi.  Now, that might not seem too accurate, but remember that the more rectangles you use, the more accurate your answer will be (you can sketch out that idea to see what I mean).  Don't forget, though, that's only half of the actual equation we're going to use.  

The other half uses a similar method called the trapezoid rule.  We're using the same number of rectangles (4) and have the same five points (0, 1/4, 1/2, 3/4, 1).  This time, instead of using the midpoint, we're going to use the average height between two points.  It's going to be roughly double the work since we're plugging in double the amount of points, and each rectangle area you find here should not be the same as each corresponding rectangle area that you found with the midpoint rule, which is a good way to check your work.  One of these values is an upper estimate, and the other is a lower estimate.  

To use the trapezoid rule, find the height at one point by plugging in the first point, then find the height at the next point by plugging in the second point to get two separate height values.  Then find the average of those heights, and multiply by the width to get the area of the first rectangle.  Then do it for the second and third points, third and fourth, etc.  That is why we needed one more point than the number of rectangles we had.  For example, we have 1 / (1 + 0^2) = 1 and 1 / (1 + (1/4)^2) = 16/17.  The average of 1 and 16/17 is 33/34.  Then 33/34 multiplied by the width (1/4) gives us the area for the first rectangle.  Do it for the other three, then add those four values up.  

Finally, we're going to take a weighted average of our midpoint rule and trapezoid rule estimations.  This is called Simpson's Rule.  Add 2 of the midpoint values to 1 of the trapezoid values, and divide by 3.  This is our final estimation for pi / 4.  The reason midpoint is more accurate is because it is the same as the trapezoid that is tangent to the middle of the curve, which has roughly half the blank space as a trapezoid made from two points at the ends of the curve.  Again, it's a little hard to see that without a picture, so don't worry too much about it.  In the end, after you multiply by 4, you really do have a value that's not too far off from pi.  

Remember that the more rectangles you make, the closer your estimations will get.  This way is the easiest to do by hand that I've found, but there are other methods too.  There is an infinite series (x^3) / 1 – (x^3) / 3 + (x^3) / 5 – (x^3) / 7…  that actually converges to pi, but you need to take it out to about 2 million terms to get an estimation that can be used accurately in calculations.  There is also a method done by measuring the lengths of sides of polygons inscribed in and circumscribed around a circle. The more sides the polygons have, the closer to each other the values of the inscribed and circumscribed polygons become, and the value they both get close to is pi.  This is actually the original way pi was calculated.  

For one last kick, I've used my resources to calculate pi using 20,000 rectangles.  This estimation is actually accurate to 11 decimal places, which is quite amazing considering it's still an estimation.  I wrote a simple C program to do the math for me, and I've included it here so that you too can calculate pi to your heart's desire.  Due to the internal precision of the computer, (the way doubles are stored) it won't get any more accurate even if you increase the amount of rectangles.  I'll bet you never knew you were so smart!  Happy Pi Day, and good luck!  
