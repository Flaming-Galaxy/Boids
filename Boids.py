import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist, cdist
from numpy.linalg import norm

# Window size
width, height = 640, 480


class Boids:

    def __init__(self, N):
        # Initialise position & velocities
        self.position = [width/2.0, height/2.0] + \
            10*np.random.rand(2*N).reshape(N, 2)

        # Normalized random velocities
        angles = 2*math.pi*np.random.rand(N)
        self.velocity = np.array(list(zip(np.sin(angles), np.cos(angles))))
        self.N = N

        # Min distance of approach
        self.minimumDistance = 25.0

        # Maximum magnitude of velocities calculated by "rules"
        self.maximumVelocityRule = 0.03

        # Maximum maginitude of final velocity
        self.maximumVelocity = 2.0

    # Updating animation each frame
    def tick(self, frameNum, body, head):
        # Get pairwise distances
        self.distanceMatrix = squareform(pdist(self.position))

        # Apply boids rules
        self.velocity += self.apply_rules()
        self.limit(self.velocity, self.maximumVelocity)
        self.position += self.velocity
        self.boundary_conditions()

        # Update data
        body.set_data(self.position.reshape(2*self.N)[::2],
                      self.position.reshape(2*self.N)[1::2])
        vec = self.position + 10*self.velocity/self.maximumVelocity
        head.set_data(vec.reshape(2*self.N)[::2],
                      vec.reshape(2*self.N)[1::2])

    # Limiting size of calculated vectors
    def limit_vectors(self, vec, maximumValue):
        magnitude = norm(vec)
        if magnitude > maximumValue:
            vec[0], vec[1] = vec[0]*maximumValue / \
                magnitude, vec[1]*maximumValue/magnitude

    # Limiting vectors in a matrix
    def limit(self, X, maximumValue):
        for vector in X:
            self.limit_vectors(vector, maximumValue)

    # Apply boundary conditions
    def boundary_conditions(self):
        # Small buffer for aesthetics
        deltaR = 2.0
        for coord in self.position:
            if coord[0] > width + deltaR:
                coord[0] = - deltaR
            if coord[0] < - deltaR:
                coord[0] = width + deltaR
            if coord[1] > height + deltaR:
                coord[1] = - deltaR
            if coord[1] < - deltaR:
                coord[1] = height + deltaR

    # Apply boids rules
    def apply_rules(self):
        # Rule 1 - Separation
        D = self.distanceMatrix < 25.0
        velocity = self.position * \
            D.sum(axis=1).reshape(self.N, 1) - D.dot(self.position)
        self.limit(velocity, self.maximumVelocityRule)

        # Different distance threshold for rules 2 and 3
        D = self.distanceMatrix < 50.0

        # Rule 2 - Alignment
        velocity2 = D.dot(self.velocity)
        self.limit(velocity2, self.maximumVelocityRule)
        velocity += velocity2

        # Rule 3 - Cohesion
        velocity3 = D.dot(self.position) - self.position
        self.limit(velocity3, self.maximumVelocityRule)
        velocity += velocity3

        return velocity


# Update each frame
def tick(frameNum, body, head, boids):
    boids.tick(frameNum, body, head)
    return body, head


# Main() function
def main():
    print('Starting boids...')

    parser = argparse.ArgumentParser(
        description="Implementing Craig Reynold's Boids...")

    # Add arguments
    parser.add_argument('--num-boids', dest='N', required=False)
    args = parser.parse_args()

    # Number of boids
    N = 100
    if args.N:
        N = int(args.N)

    # Create boids
    boids = Boids(N)

    # Setup plot
    figure = plt.figure()
    axes = plt.axes(xlim=(0, width), ylim=(0, height))

    body, = axes.plot([], [], markersize=10,
                      c='k', marker='o', ls='None')
    head, = axes.plot([], [], markersize=4,
                      c='r', marker='o', ls='None')
    anim = animation.FuncAnimation(figure, tick, fargs=(body, head, boids),
                                   interval=50)

    plt.show()


# Call main() function when file is run
if __name__ == '__main__':
    main()
