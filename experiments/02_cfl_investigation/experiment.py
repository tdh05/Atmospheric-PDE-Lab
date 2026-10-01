import numpy as np
import matplotlib.pyplot as plt 

from atmospheric_pde.initial_conditions import gaussian 
from atmospheric_pde.advection import analytical_periodic_advection
from atmospheric_pde.solver import solve_upwind

L = 10.0
nx = 200

x = np.linspace(0.0, L, nx, endpoint=False)
dx = L / nx

centre = 2.0
width = 0.3
amplitude = 1.0

u = 1.0

target_time = 2.0
cfl_values = [0.1, 0.25, 0.5, 0.75, 1.0]

K = 3

q0 = gaussian(x, centre, width, amplitude)

results = []

for cfl in cfl_values:
    
    dt = cfl * dx / u
    
    n_steps = int(np.ceil(target_time / dt ))
    true_time = n_steps * dt
    
    q_numerical = solve_upwind(q0, u, dt, dx, n_steps)
    q_exact = analytical_periodic_advection(x, u, true_time, centre, width, amplitude, L, K)
    
    error = q_numerical - q_exact
    
    L_2_error = np.sqrt(np.mean(error**2))
    L_inf_error = np.max(np.abs(error))
    
    results.append(
        (cfl, dt, true_time, L_2_error, L_inf_error)
    )

cfls = np.array([result[0] for result in results])
L_2_errors = np.array([result[3] for result in results])
L_inf_errors = np.array([result[4] for result in results])

for cfl, dt, true_time, L2, Linf in results:
    print(
        f"CFL = {cfl:.2f} | "
        f"dt = {dt:.5f} | "
        f"time = {true_time:.3f} | "
        f"L2 = {L2:.6f} | "
        f"Linf = {Linf:.6f}"
    )

fig, ax = plt.subplots()

ax.plot(cfls, L_2_errors, marker="o", label="L2_error")
ax.plot(cfls, L_inf_errors, marker="o", label="Linf_error")

ax.set_xlabel("CFL number")
ax.set_ylabel("Error")
ax.set_title("Upwind Accuracy as a Function of CFL Number")
ax.legend()

plt.show()