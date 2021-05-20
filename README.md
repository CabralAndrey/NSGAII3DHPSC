# NSGAII3DHPSC
Repository containing the code used and develloped for the 3DHPSC protein model for biobjective  landscape, using the Non-dominated Sorting Genetic Algorithm II(NSGA -II) method. 

this project is in constant development for improvements in execution and results.

This project was used as a theme for my master's degree.

For explanations about the algorithm and movement of the protein in the truss please read chapter 3 of the dissertation.

This project use Creative Commons Attribution 4.0 for license.

A code for a developing version of SPEA II is available for evaluation and testing, but it still proves to be computationally infeasible.


-------------------------------------------------------------------------------

This project use the python 3.5 version and the follow packages:

- Numpy
- Math
- Random (Use Mersenne Twister )
- matplotlib (For plot the protein structure - commented)
- Numba(commented because we do not use it in the final execution, but it is implemented for tests.)
- Decimal (dammm....)
- time
- statistics(damm...[2])
 

up to line 668 are the functions needed for the algorithm to work. They are described in the dissertation.

Note that on line 582 to line 591 is commented out for testing. Uncomment and comment on line 596 in case of using an external file with the sequence you are going to test.

From line 681 to 1001 there is the NSGA II with the changes to run the 3000 generations of the protein, as well as the functions and function calls.

in the line 1108 use this part, only for plot the best protein after the execution.

The sequences and results are available in the dissertation.
