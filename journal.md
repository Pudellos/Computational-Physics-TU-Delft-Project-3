# Week 1
We have implemented an algorithm to solve the Lindblad equation for a spin 1/2 system, with relaxation and decoherence, using a sort-off Euler method. In the relaxation plot we start off in the up state and over time, due to relaxation and excitation processes, these slowly decay to 1/2. This is because the up->down process has same rate as the down->up process. In the decohernce plot we start of in the plus state (up + down) and this state decays to 0, in an oscillatory fashion, which is what we expect to happen! Next step is to upgrade to more complex state, such as the entangled state and the coherent state of light. We could also maybe implement Verlet method instead of Euler method to solve the equation. 

![alt text](Figures/relaxation_plot.svg "Title Text")
![alt text](Figures/decoherence_plot.svg "Title Text")

# Week 2
This week we fitted the curves of the relaxation plot and show that the up->down process has the same rate as the down->up process namely $\gamma= 0.200$.

![alt text](Figures/rho00fit.png "Title Text")
![alt text](Figures/rho11fit.png "Title Text")