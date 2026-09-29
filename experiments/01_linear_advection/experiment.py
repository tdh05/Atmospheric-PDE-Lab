import numpy as np
import matplotlib.pyplot as plt 

from atmospheric_pde.initial_conditions import gaussian 
from atmospheric_pde.advection import analytical_periodic_advection
from atmospheric_pde.solver import solve_upwind

L = 10.0
nx = 200

x = np.linspace(0.0, L, nx, endpoint=False)

centre = 2.0
width = 0.3
amplitude = 1.0

u = 1.0
cfl = 0.5

t_final = 2.0
K = 3

q0 = gaussian(x, centre, width, amplitude)

dx = L / nx

dt = (cfl * dx) / u

n_steps = int(np.ceil(t_final / dt))

true_t_final = n_steps * dt

q_numerical = solve_upwind(q0, u, dt, dx, n_steps)
q_exact = analytical_periodic_advection(x, u, true_t_final, centre, width, amplitude, L, K)

error = q_numerical - q_exact

L_2_error = np.sqrt(np.mean(error**2))
L_inf_error = np.max(np.abs(error))

print(f"CFL: {cfl}")
print(f"dx : {dx}")
print(f"dt : {dt}")
print(f"Number of steps : {n_steps}")
print(f"True final time : {true_t_final}")
print(f"L2 error : {L_2_error}")
print(f"Linf error : {L_inf_error}")

fig, ax = plt.subplots()

ax.plot(x, q0, label="Initial")
ax.plot(x, q_exact, label="Analytical")
ax.plot(x, q_numerical, label="Upwind")

ax.set_xlabel("x")
ax.set_ylabel("Tracer Concentration")
ax.set_title("Linear Advection: Numerical Diffusion in the Upwind Scheme")
ax.legend()

plt.show()