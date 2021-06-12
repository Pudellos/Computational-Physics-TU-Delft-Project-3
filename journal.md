# Week 1
We have implemented an algorithm to solve the Lindblad equation for a spin 1/2 system, with relaxation and decoherence, using a sort-off Euler method. In the relaxation plot we start off in the up state and over time, due to relaxation and excitation processes, these slowly decay to 1/2. This is because the up->down process has same rate as the down->up process. In the decohernce plot we start of in the plus state (up + down) and this state decays to 0, in an oscillatory fashion, which is what we expect to happen! Next step is to upgrade to more complex state, such as the entangled state and the coherent state of light. We could also maybe implement Verlet method instead of Euler method to solve the equation. 

![alt text](Figures/relaxation_plot.svg "Title Text")
![alt text](Figures/decoherence_plot.svg "Title Text")

# Week 2
This week we fitted the curves of the relaxation plot and show that the up->down process has the same rate as the down->up process namely $`\gamma=0.200`$.

![alt text](Figures/rho00fit.png "Title Text")
![alt text](Figures/rho11fit.png "Title Text")

We also implemented a spin echo experiment. The spin echo experiment is as follows: $`R_{\hat{x}}(\pi/2) \rightarrow R_{\hat{x}}(\pi) \rightarrow R_{\hat{x}}(\pi/2)`$. Between the pulses we wait a time $`\tau`$, here the system evolves according to the Lindblad equation. In the Figure below we have varied $`\tau`$ and plotted the fidelity of the state without the spin echo ($`F_A`$) and with the spin echo ($`F_B`$). As expected, the fidelity of the state with echo is consistently better than without, which is the goal of the spin echo.

![alt text](Figures/spin_echo.svg "Title Text")

We also extended the code to solve a coupled two spin 1/2 system. We plotted the fidelity, purity and probability to have both spins in the up-state.

![alt text](Figures/entangled_states.svg "Title Text")

The code has also been restructured. We made a State class, which contains all the information of the state of our system and can be used to calculate internally any observable we give it. It can also calculate the probability to measure any state if you give it the corresponding measurement operator. In main we can now easily experiment with the Lindblad equation and look at different input parameters!

# Week 3
The next phenomenon we simulated is the Ramsey experiment, the exact details are quite technical but the idea is that we rotate a particle (in the up or down state) with a short pulse of $\pi/2$ and the again with a pulse of $-\pi/2$. If we then look at the decay times and oscillation frequencies of $P_{down}$, $\rho_{00}$ and $\frac{1}{2}(\rho_{01}+\rho_{10})$ we can find T1 and T2 of the system. This is almost exaclty what happens in an NMR. To simulate this the Lindblad equation was solved for each timestep (250 times) which took 122.46 s, the data was then fitted to curves which we found in the slides of the course REFERENCE NEEDED. The values we find are (on average): $\omega_0=1$, $T2=20.74$, $T1=10$ and $a_{\beta}=1$.


![alt text](Figures/ramsey.svg "Title Text")
![alt text](Figures/Pdown.svg "Title Text")
![alt text](Figures/mixed.svg "Title Text")
![alt text](Figures/updown.svg "Title Text")
