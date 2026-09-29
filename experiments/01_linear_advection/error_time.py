import numpy as np
import matplotlib.pyplot as plt 

from atmospheric_pde.initial_conditions import gaussian 
from atmospheric_pde.advection import analytical_periodic_advection
from atmospheric_pde.solver import solve_upwind

L = 10.0
nx = 200

x = np.linspace(0.0, L, nx, endpoint=False)
target_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

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

true_times = []

L_2_errors = []
L_inf_errors = []

for time in target_times:
    n_steps = int(np.ceil(time / dt))
    
    true_time = n_steps * dt
    
    q_numerical = solve_upwind(q0, u, dt, dx, n_steps)
    q_exact = analytical_periodic_advection(x, u, true_time, centre, width, amplitude, L, K)
    
    error = q_numerical - q_exact
    
    L_2_error = np.sqrt(np.mean(error**2))
    L_inf_error = np.max(np.abs(error))
    
    true_times.append(true_time)
    L_2_errors.append(L_2_error)
    L_inf_errors.append(L_inf_error)
    
true_times = np.array(true_times)
L_2_errors = np.array(L_2_errors)
L_inf_errors = np.array(L_inf_errors)

print(f"CFL: {cfl}")
print(f"dx : {dx}")
print(f"dt : {dt}")
print()

for time, L_2, L_inf in zip(true_times, L_2_errors, L_inf_errors):
    print(f"Time : {time:.3f} | "
          f"L_2 Error : {L_2:.6f} | "
          f"L_inf Error : {L_inf:.6f}")
    
fig, ax = plt.subplots()

ax.plot(true_times, L_2_errors, marker="o", label="L_2 Error")

ax.plot(true_times, L_inf_errors, marker="o", label="L_inf Error")

ax.set_xlabel("Time")
ax.set_ylabel("Error")

ax.set_title("Error Growth with transport Time: Upwind Scheme")

ax.legend()

plt.show()
