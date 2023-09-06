import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Acircle/Asquare = pi/4 so we can approximate pi by generating random points
# inside the square and calculating the ratio between those areas.
points = 1000000
rand = 2*np.random.rand(2*points) - 1
randpoints = rand.reshape(points, 2) # Reshape them so it generates pairs of numbers
normpoints = randpoints[:, 0]**2 + randpoints[:, 1]**2  # Equation of the circle (x^2+y^2)
pointsOut = randpoints[normpoints > 1] # If ith element of normpoints is >1 then ith element of
# randpoints is inserted in pointOut
pointsIn = randpoints[normpoints <= 1] # The same as pointsOut
piapprox = 4 * len(pointsIn)/(len(pointsOut) + len(pointsIn))
print(piapprox)