import logging
from pathlib import Path

import numpy as np
from numpy import pi

import capytaine as cpt


import matplotlib.pyplot as plt
import matplotlib.animation as animation

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s: %(message)s')

bem_solver = cpt.BEMSolver()


#physical parameters
rho_water = 1023.
rho_ice = 920.
g = 9.81

width = 5.
thickness = 50.
height = 1.
#draft from buoyancy contrain
draft = height * rho_ice/rho_water
#vertical coordinate of barycenter
zc = -draft + height/2

print(zc)

#mass from buoyancy balance
mass = width * thickness * draft * rho_water

#hydrostatic stiffness from geometry
hyd_stiff = width * thickness * g * rho_water


body = cpt.RectangularParallelepiped(size=(width, thickness, height),
                    center=(0., 0., zc), resolution=(5, 20, 8))


body.add_translation_dof(name='Heave')

body.keep_immersed_part()


ny = 3
L = 25
nx = 100
fs = cpt.FreeSurface(x_range=(-L, L), y_range=(-1, 1), nx=nx, ny=ny)
x = np.linspace(-L, L, nx)

wave_direction = 0.0
omega = 1.5
wave_amplitude = 1.

# SOLVE BEM PROBLEMS
problems = [cpt.RadiationProblem(omega=omega, body=body, radiating_dof=dof) for dof in body.dofs]
problems += [cpt.DiffractionProblem(omega=omega, body=body, wave_direction=wave_direction)]
results = [bem_solver.solve(problem) for problem in problems]
*radiation_results, diffraction_result = results
data = cpt.assemble_dataset(results)

# extract data
added_mass = data.added_mass.data[0,:,:]
radiation_damping = data.radiation_damping.data[0,:,:]
Froude_Krylov_force = data.Froude_Krylov_force.data[0,:,:].T
diffraction_force = data.diffraction_force.data[0,:,:].T

excitation_force = Froude_Krylov_force + diffraction_force

# impedance computation
LHS = (-omega**2 * (added_mass + mass) - 1j*omega*radiation_damping
        + hyd_stiff)
rao =  excitation_force/LHS

#turn dict keys into list of dof names
dofs_names = list(body.dofs.keys())
rao_faces_motion = sum(rao[ii] *
            body.full_body.dofs[dofs_names[ii]] for ii in range(body.nb_dofs))


# COMPUTE FREE SURFACE ELEVATION
# Compute the diffracted wave pattern
diffraction_elevation = bem_solver.get_free_surface_elevation(diffraction_result, fs)
incoming_waves_elevation = fs.incoming_waves(diffraction_result)

# Compute the wave pattern radiated by the RAO
radiation_elevations_per_dof = {res.radiating_dof: bem_solver.get_free_surface_elevation(res, fs) for res in radiation_results}
rao_radiation_elevation = sum(rao[ii] *
            radiation_elevations_per_dof[dofs_names[ii]] for ii in range(body.nb_dofs))

diffraction_elevation = diffraction_elevation.reshape(nx, ny).T
incoming_waves_elevation = incoming_waves_elevation.reshape(nx, ny).T
rao_radiation_elevation = rao_radiation_elevation.reshape(nx, ny).T


def incident_wave(t):
    return np.real(incoming_waves_elevation[1,:] * np.exp(-1j*omega*t))

def diffracted_wave(t):
    return np.real(diffraction_elevation[1,:] * np.exp(-1j*omega*t))

def radiated_wave(t):
    return np.real(rao_radiation_elevation[1,:]*(-1j*omega) * np.exp(-1j*omega*t))

def body_position(t):
    return np.real(rao * np.exp(-1j*omega*t))


fig, ax = plt.subplots()
#line1, = ax.plot(x, incident_wave(0), '--')
#line2, = ax.plot(x, diffracted_wave(0), '--')
#line2a, = ax.plot(x, diffracted_wave(0)+incident_wave(0), 'r')
line3, = ax.plot(x, radiated_wave(0), 'g')
#line4, = ax.plot(x, incident_wave(0) + diffracted_wave(0) + radiated_wave(0))
body_point, = ax.plot([-width/2, width/2], [body_position(0)[0][0], body_position(0)[0][0]], 'k-')
#body_point, = ax.plot([-width/2, width/2], [0,0], 'k-')
ax.set_ylim(-2, 2)

nframes = 120

def animate(i):
    delta_t = (2*np.pi/omega) / nframes
    t = i*delta_t
    #line1.set_ydata(incident_wave(t))
    #line2.set_ydata(diffracted_wave(t))
    #line2a.set_ydata(diffracted_wave(t)+incident_wave(t))
    line3.set_ydata(radiated_wave(t))
    #line4.set_ydata(incident_wave(t) + diffracted_wave(t) + radiated_wave(t))
    body_point.set_ydata([body_position(t)[0][0], body_position(t)[0][0]])


    #return line1, line2, line2a,# line4, body_point,
    return line3, body_point,

ani = animation.FuncAnimation(
    fig, animate, interval=20, frames=nframes, blit=True, save_count=5)


ani.save('incident.gif', writer='imagemagick', fps=24)

plt.show()
