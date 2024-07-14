import numpy as np

class EinsteinHelfandViscosity:
    def __init__(self, positions, momenta, temperature, volume, boltzmann_constant=1.380649e-23):
        self.positions = np.array(positions)
        self.momenta = np.array(momenta)
        self.temperature = temperature
        self.volume = volume
        self.boltzmann_constant = boltzmann_constant
        self.num_particles = self.positions.shape[1]

    def calculate_helfand_moment(self, t):
        H = np.zeros((t, 3, 3))
        for i in range(self.num_particles):
            for alpha in range(3):
                for beta in range(3):
                    H[:, alpha, beta] += self.positions[:, i, alpha] * self.momenta[:, i, beta]
        return H

    def calculate_viscosity(self):
        num_timesteps = self.positions.shape[0]
        H = self.calculate_helfand_moment(num_timesteps)
        H_diff = H - H[0, np.newaxis, :, :]
        H_diff_squared = H_diff**2
        H_diff_squared_sum = np.sum(H_diff_squared, axis=(1, 2))
        viscosity = (1 / (2 * self.boltzmann_constant * self.temperature * self.volume)) * np.mean(H_diff_squared_sum)
        return viscosity

# Example usage:
# positions and momenta should be numpy arrays of shape (num_timesteps, num_particles, 3)
positions = np.random.rand(100, 10, 3)
momenta = np.random.rand(100, 10, 3)
temperature = 300  # in Kelvin
volume = 1e-24  # in m^3

viscosity_calculator = EinsteinHelfandViscosity(positions, momenta, temperature, volume)
viscosity = viscosity_calculator.calculate_viscosity()
print(f"Calculated Viscosity: {viscosity}")
