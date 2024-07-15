import matplotlib.pyplot as plt

class EinsteinHelfandViscosity:
    def __init__(self, positions, momenta, temperature, volume, boltzmann_constant=1.380649e-23):
        self.positions = np.array(positions)
        self.momenta = np.array(momenta)
        self.temperature = temperature
        self.volume = volume
        self.boltzmann_constant = boltzmann_constant
        self.num_particles = self.positions.shape[1]

    def calculate_helfand_moment(self):
        num_timesteps = self.positions.shape[0]
        H = np.zeros((num_timesteps, 3, 3))
        for t in range(num_timesteps):
            for i in range(self.num_particles):
                for alpha in range(3):
                    for beta in range(3):
                        H[t, alpha, beta] += self.positions[t, i, alpha] * self.momenta[t, i, beta]
        return H

    def calculate_viscosity(self):
        H = self.calculate_helfand_moment()
        H_diff = H - H[0, np.newaxis, :, :]
        H_diff_squared = H_diff**2
        H_diff_squared_sum = np.sum(H_diff_squared, axis=(1, 2))
        viscosity = (1 / (2 * self.boltzmann_constant * self.temperature * self.volume)) * np.mean(H_diff_squared_sum)
        return viscosity, H_diff_squared_sum

# Example usage:
positions = np.random.rand(100, 10, 3)
momenta = np.random.rand(100, 10, 3)
temperature = 300  # in Kelvin
volume = 1e-24  # in m^3

viscosity_calculator = EinsteinHelfandViscosity(positions, momenta, temperature, volume)
viscosity, H_diff_squared_sum = viscosity_calculator.calculate_viscosity()
print(f"Calculated Viscosity: {viscosity}")

# Plotting Helfand Moment Differences (H(t) - H(0))
H = viscosity_calculator.calculate_helfand_moment()
H_diff = H - H[0, np.newaxis, :, :]
time_steps = np.arange(H_diff.shape[0])

for alpha in range(3):
    for beta in range(3):
        plt.plot(time_steps, H_diff[:, alpha, beta], label=f'H_diff_{alpha}{beta}')

plt.xlabel('Time Step')
plt.ylabel('Helfand Moment Difference')
plt.legend()
plt.title('Helfand Moment Differences Over Time')
plt.show()

# Plotting Mean Squared Displacement of Helfand Moment Differences
plt.plot(time_steps, H_diff_squared_sum)
plt.xlabel('Time Step')
plt.ylabel('MSD of Helfand Moment')
plt.title('Mean Squared Displacement of Helfand Moment Over Time')
plt.show()
