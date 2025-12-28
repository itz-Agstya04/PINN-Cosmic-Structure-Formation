import numpy as np


class PIC1D:
    """
    Minimal 1D Particle-In-Cell (PIC) solver for Vlasov–Poisson.
    Used as a classical numerical baseline for comparison with PINNs.
    """

    def __init__(self, Np=5000, Nx=64, L=2.0, dt=0.05):
        self.Np = Np        # number of particles
        self.Nx = Nx        # number of grid points
        self.L = L          # domain length
        self.dt = dt        # time step

        # Particle positions and velocities
        self.x = np.random.uniform(-L / 2, L / 2, Np)
        self.v = np.random.normal(0.0, 1.0, Np)

        # Grid spacing
        self.dx = L / Nx

        # Grid quantities
        self.rho = np.zeros(Nx)
        self.phi = np.zeros(Nx)
        self.E = np.zeros(Nx)

    def deposit_charge(self):
        """
        Deposit particle charge onto grid (nearest-grid-point scheme).
        """
        self.rho.fill(0.0)

        indices = ((self.x + self.L / 2) / self.dx).astype(int) % self.Nx
        for i in indices:
            self.rho[i] += 1.0

        # Normalize by grid spacing
        self.rho /= self.dx

    def solve_poisson(self, iterations=100):
        """
        Solve Poisson equation using simple Jacobi iteration.
        """
        self.phi.fill(0.0)

        for _ in range(iterations):
            self.phi[1:-1] = 0.5 * (
                self.phi[:-2] + self.phi[2:] - self.dx**2 * self.rho[1:-1]
            )

        # Electric field (central difference)
        self.E[1:-1] = -(self.phi[2:] - self.phi[:-2]) / (2 * self.dx)

    def push_particles(self):
        """
        Advance particles using leapfrog-style update.
        """
        indices = ((self.x + self.L / 2) / self.dx).astype(int) % self.Nx

        self.v += self.E[indices] * self.dt
        self.x += self.v * self.dt

        # Periodic boundary conditions
        self.x = (self.x + self.L / 2) % self.L - self.L / 2

    def step(self):
        """
        Perform one PIC time step.
        """
        self.deposit_charge()
        self.solve_poisson()
        self.push_particles()
