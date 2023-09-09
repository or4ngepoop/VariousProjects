import numpy as np
import matplotlib.pyplot as plt

# Starting configuration
length = 20
num = length ** 2

# We have to generate random orientations of the magnetic momentum of it's magnetic dipole
randphi = 2*np.pi * np.random.rand(num).reshape (length, length, 1)
randtheta = np.pi * np.random.rand(num).reshape (length, length, 1)
mag = np.array([ np.cos(randphi)*np.sin(randtheta), np.sin(randphi)*np.sin(randtheta), np.cos(randtheta) ])
coords = np.array(np.meshgrid(np.arange(length),
                              np.arange(length),
                              np.arange(1)))
# Visualization
'''
plt.rcParams['figure.figsize'] = [40, 10]
arrowplot = plt.axes(projection='3d')
arrowplot.set_zlim(-1.5, 1.5)
arrowplot.set_box_aspect(aspect=(length, length, 3))
arrowplot.axis(False)
arrowplot.quiver(coords[0], coords[1], coords[2],
                 mag[0], mag[1], mag[2])
arrowplot.scatter3D(coords[0], coords[1], coords[2], color='Red')

plt.show()
'''

# Calculating the energy
J = 1


# The energy of each dipole is only affected by its neighbors
# The modulus was inserted to prevent errors on the boundaries
def energyExchangeContribution(mag, x, y):
    return -0.5 * J * np.dot( mag[:,x,y,0], mag[:,(x+1)%length,y,0]+mag[:,(x-1)%length,y,0]+mag[:,x,(y+1)%length,0]+mag[:,x,(y-1)%length,0])


def energyExchange(mag):
    # mag: Array magnetic moments
    energy = 0
    for x in range(length):
        for y in range(length):
            energy = energy + energyExchangeContribution(mag,x,y)
    return energy


#  Update 2: We apply an external magnetic field
mu = 1
B = np.array([0, 0, 0])

def energyMagneticContribution(mag,x,y):
    return - mu * np.dot(B, mag[:,x,y,0])

def energyMagnetic(mag):
    # mag: Array magnetic moments
    energy = 0
    for x in range(length):
        for y in range(length):
            energy = energy + energyMagneticContribution(mag,x,y)
    return energy


# Update 3: We include the Dzyaloshinskii–Moriya interaction (asymmetric exchange).
D = 0.3

def energyDMIContribution(mag,x,y):
    right = mag[1,x,y,0]*mag[2,(x+1)%length,y,0] - mag[2,x,y,0]*mag[1,(x+1)%length,y,0]
    left = -mag[1,x,y,0]*mag[2,(x-1)%length,y,0] + mag[2,x,y,0]*mag[1,(x-1)%length,y,0]
    up = mag[2,x,y,0]*mag[0,x,(y+1)%length,0] - mag[0,x,y,0]*mag[2,x,(y+1)%length,0]
    down = -mag[2,x,y,0]*mag[0,x,(y-1)%length,0] + mag[0,x,y,0]*mag[2,x,(y-1)%length,0]
    return 0.5 * D * (right + left + up + down)

def energyDMI(mag):
    # mag: Array magnetic moments
    energy = 0
    for x in range(length):
        for y in range(length):
            energy = energy + energyDMIContribution(mag,x,y)
    return energy


# Metropolis steps to define state of least energy
def stepExchange(mag):
    # 1.
    x = np.random.randint(length)
    y = np.random.randint(length)
    energyold = 2 * energyExchangeContribution(mag,x,y)
    # 2.
    randphi = 2*np.pi * np.random.rand()
    randtheta = np.pi * np.random.rand()
    # Problem with the ids: savemag = mag[:,x,y,0] ==> savemag will be changed once mag[:,x,y,0] is changed in the next line
    savemag = np.array(mag[:,x,y,0])
    mag[:,x,y,0] = np.array([ np.cos(randphi)*np.sin(randtheta), np.sin(randphi)*np.sin(randtheta), np.cos(randtheta) ])
    # 3.
    energynew = 2 * energyExchangeContribution(mag,x,y)
    # 4.
    if ( energynew < energyold):
        # accept the change & update the energy
        energychange = energynew - energyold
    else:
        # decline & restore old moment
        mag[:,x,y,0] = savemag
        energychange = 0
    return [mag, energychange]


def stepT(mag, kBtemp):
    # 1.
    x = np.random.randint(length)
    y = np.random.randint(length)
    energyold = 2 * energyExchangeContribution(mag,x,y) + energyMagneticContribution(mag,x,y) + 2 * energyDMIContribution(mag,x,y)
    # 2.
    randphi = 2*np.pi * np.random.rand()
    randtheta = np.pi * np.random.rand()
    # Problem with the ids: savemag = mag[:,x,y,0] ==> savemag will be changed once mag[:,x,y,0] is changed in the next line
    savemag = np.array(mag[:,x,y,0])
    mag[:,x,y,0] = np.array([ np.cos(randphi)*np.sin(randtheta), np.sin(randphi)*np.sin(randtheta), np.cos(randtheta) ])
    # 3.
    energynew = 2 * energyExchangeContribution(mag,x,y) + energyMagneticContribution(mag,x,y) + 2 * energyDMIContribution(mag,x,y)
    # 4.
    if ( energynew < energyold):
        # accept the change & update the energy
        energychange = energynew - energyold
    else:
        if np.random.rand() < np.exp( -(energynew - energyold) / kBtemp ):
            # accept the change & update the energy
            energychange = energynew - energyold
        else:
            # decline & restore old moment
            mag[:,x,y,0] = savemag
            energychange = 0
    return [mag, energychange]


numberSteps = 1000000

#optional
# energy = energyExchange(mag)
energy = energyExchange(mag) + energyMagnetic(mag) + energyDMI(mag)
energyList = [energy]

for i in range(numberSteps):
    # mag, energychange = stepExchange(mag)          # Initial version: zero temperature
    kBtemp = 0.2*(1-i/numberSteps)
    # mag, energychange = stepExchangeT(mag,kBtemp)  # Update 1: Consider finite temperatures
    mag, energychange = stepT(mag,kBtemp)            # Update 2&3: Add interaction with magnetic field & DMI
    #optional
    energy = energy + energychange
    energyList.append(energy)

#plt.plot(range(numberSteps+1), energyList)
plt.rcParams['figure.figsize'] = [40, 10]
arrowplot = plt.axes(projection='3d')
arrowplot.set_zlim(-1.5, 1.5)
arrowplot.set_box_aspect(aspect=(length, length, 3))
arrowplot.axis(False)
arrowplot.quiver(coords[0], coords[1], coords[2],
                 mag[0], mag[1], mag[2])
arrowplot.scatter3D(coords[0], coords[1], coords[2], color='Red')

plt.show()

