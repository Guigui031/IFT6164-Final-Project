- Create environmnent file with all the dependencies.
- Run baseline MAPPO with no perturbation: with and without param sharing.
- Devise basic attacks on observations: Random noise and FGSM.
- Plot the results 

Hypothesis:
If parameters across agents are shared, perturbation in the env will crash the whole system -> Coop Multi Agent will get less rewards, than independently trained models. 

The logic is that a well devised attack can lead to catastrophic results in shared setting. You need to devise a single attack to crash the system, vs devising a distinct attack per agent. 

We perturb the observations only, so we can use a centralised critic in any case. The ablation study is done on the shared vs non-shared parameters. 


