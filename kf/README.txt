description of versions:
v1: 400 * 1 matrix, every object can only occur 1 time per frame 
v2: 8000 * 1 matrix, every object can occur 20 times per frame, too slow 
v3: 840 * 1 matrix, at most 10 objects per frame, has full prob distribution, (may update using the wrong recognition)
v4: 400N * 1 matrix, every object can occur at most N times per frame (dynamic)


time used: 
400 matrix:  0.0331721379998271  --> 20 fps
800 matrix:  0.14566815299986047  --> 6 fps
1200 matrix:  0.33988786400004756  --> 3 fps
1600 matrix:  0.7434118329992998  --> 1 fps
2000 matrix:  2.7712933889997657  --> 0.3 fps
8000 matrix:  107.98598215899983  --> the library is twice as fast as my own code 

