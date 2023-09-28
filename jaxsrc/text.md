This coupled system of PDEs differs slightly from the HJ PDE~\eqref{eqt:cont_HJ_eqt}, with the distinction that the first row entails an inequality rather than an equality. As indicated by the final row, if $\rho(x,t)\neq 0$ for all $x,t$, the initial inequality in the first row transforms into an equality, thus yielding a solution to the HJ PDE~\eqref{eqt:cont_HJ_eqt}. 
The difference between the first row and the HJ PDE~\eqref{eqt:cont_HJ_eqt} is because we added a constraint $\rho \geq 0$ in the saddle point problem [...]. For more details, see [...]. 



Here the Hamiltonian is not smooth.
We use the same method as in the first example to compute the error tables. The error table in one-dimension is shown in Table~\ref{tab:eg0_1d}, while the error table in two-dimensional is shown in Table~\ref{tab:eg0_2d}. 
The errors in the first row of both two error tables indicate the HJ PDE residual errors, which are all below $10^{-6}$. These errors show that the PDHG method converges in the sense that the HJ PDE residual drops below the threshold that we set ($10^{-6}$).
The second row of the error table shows the $\ell^1$ relative error compared to the reference solution. We observe that these errors decay to almost a half when we increase the grid size by a factor of 2, which is consistent with the fact that we are using a first order Enquist-Osher scheme.


We show the solution computed by our proposed method in Fig~\ref{fig:eg2}.
In Fig~\ref{fig:eg2} (a), we show the level sets of the one-dimensional solution solved using a larger time step $\Delta t = 0.25$, while in Fig~\ref{fig:eg2} (b), we show the level sets of the one-dimensional solution solved using a smaller time step $\Delta t = 0.025$. The two-dimensional solution solved with a larger time step $\Delta t = 0.25$ is shown in Fig~\ref{fig:eg2} (c)-(d). (c) shows the solution at time $t=0.25$, and (d) shows the solution at time $t=0.5$.


Hamilton-Jacobi (HJ) partial differential equations (PDEs) find applications in various fields such as physics \cite{Arnold1989Math, Caratheodory1965CalculusI,Caratheodory1967CalculusII,Courant1989Methods,landau1978course},
optimal control \cite{Bardi1997Optimal, Elliott1987Viscosity,fleming1976deterministic,fleming2006controlled,mceneaney2006max}, game theory \cite{BARRON1984213, Buckdahn2011Recent, Evans1984Differential, Ishii1988Representation}, and imaging sciences \cite{darbon2015convex,darbon2019decomposition,Darbon2016Algorithms}. [TODO: add mfg]
In the literature, there are different methods for solving HJ PDEs numerically. In low dimensions, high-order grid-based schemes are often applied, such as ENO, WENO, DG. In high dimensions, different methods are proposed to mitigate the curse of dimensionality, such as [...]. 

In this paper, we proposed an optimization-based method for solving HJ PDEs. We formulate the HJ PDE as a saddle point problem and then solve it using the primal-dual hybrid gradient (PDHG) method \cite{Chambolle2011First}. Compared with other optimization-based method [...], our method can handle a larger class of Hamiltonian functions, which may be non-smooth and may depend on $(x,t)$. Moreover, due to the simple formulation of the saddle point problem, our algorithm have partial theoretical guarantees. 
Compared with grid-based methods, although our method is first-order, we apply implicit time discretization, which is unconditionally stable, which makes it possible to choose a large time step.

[literature about pdhg solving pdes]



Hamilton-Jacobi (HJ) partial differential equations (PDEs) find applications in various fields such as physics \cite{Arnold1989Math, Caratheodory1965CalculusI,Caratheodory1967CalculusII,Courant1989Methods,landau1978course},
optimal control \cite{Bardi1997Optimal, Elliott1987Viscosity,fleming1976deterministic,fleming2006controlled,mceneaney2006max}, game theory \cite{BARRON1984213, Buckdahn2011Recent, Evans1984Differential, Ishii1988Representation}, and imaging sciences \cite{darbon2015convex,darbon2019decomposition,Darbon2016Algorithms,darbon2022hamilton}. [TODO: add mfg]
In existing literature, a variety of approaches have been explored to numerically address Hamilton-Jacobi partial differential equations (HJ PDEs). In lower dimensions, sophisticated grid-based techniques like ENO \cite{Osher1991High}, WENO \cite{Jiang2000Weighted}, and DG \cite{Hu1999Discontinuous} are commonly employed, while in higher dimensions, strategies have been proposed to manage the challenges arising from the curse of dimensionality.
These works include, but are not limited to, max-plus algebra methods \cite{mceneaney2006max,akian2006max,akian2008max, dower2015max,Fleming2000Max,gaubert2011curse,McEneaney2007COD,mceneaney2008curse,mceneaney2009convergence}, dynamic programming and reinforcement learning \cite{alla2019efficient,bertsekas2019reinforcement}, tensor decomposition techniques \cite{dolgov2019tensor,horowitz2014linear,todorov2009efficient}, sparse grids \cite{bokanowski2013adaptive,garcke2017suboptimal,kang2017mitigating}, model order reduction \cite{alla2017error,kunisch2004hjb}, polynomial approximation \cite{kalise2019robust,kalise2018polynomial}, optimization methods \cite{darbon2015convex,darbon2019decomposition,Darbon2016Algorithms,yegorov2017perspectives,chow2019algorithm,chen2021hopf,chen2021lax} and neural networks \cite{darbon2019overcoming,bachouch2018deep, Djeridane2006Neural,jiang2016using, Han2018Solving, hure2018deep, hure2019some, lambrianides2019new, Niarchos2006Neural, reisinger2019rectified,royo2016recursive, Sirignano2018DGM,darbon2021some,darbon2023neural}. 

In this study, we introduce an innovative optimization-based methodology for tackling HJ PDEs whose Hamiltonian $H(x,t,p)$ is convex with respect to $p$. Our approach formulates the HJ PDE as a saddle point problem and employs the primal-dual hybrid gradient (PDHG) method \cite{chambolle2011first} for its solution. 
Compared to other grid-based methods, although our approach has a first-order accuracy level, by leveraging implicit time discretization, we achieve unconditional stability. This characteristic enables us to adopt larger time steps.
Compared to other optimization-based methods~\cite{darbon2015convex,darbon2019decomposition,Darbon2016Algorithms,yegorov2017perspectives,chow2019algorithm,chen2021hopf,chen2021lax}, our technique boasts the capability to handle a broader range of Hamiltonian functions, including those that exhibit non-smooth behavior and dependence on both $(x,t)$. Furthermore, our algorithm benefits from its straightforward saddle point formulation, which affords partial theoretical assurances.

We show several numerical examples in one dimension and two dimensions. These numerical examples show the ability of this method to handle certain Hamiltonians which may depend on $(x,t)$. In each iteration, the updates of the functions are independent on each point, which makes it possible to use parallel computing to accelerate the algorithm. Moreover, in the special case when the Hamiltonian $H(x,t,p)$ is $1$-homogeneous in $p$, the algorithm has a simpler form, and we obtain an explicit formula for the updates of the dual variables. 

The codes are available at [...].


The paper is organized as follows. In Sec~ , we show the continuous formulation of the saddle point problem related to the HJ PDE and the corresponding PDHG algorithm in the function space. In Sec..., we focus on the one-dimensional case and show both semi-discrete and fully-discrete formulations of the algorithm. In Sec..., we show the two-dimensional case. In Sec..., we show several numerical results which demonstrate the ability of the algorithm to handle certain Hamiltonians which may depend on $(x,t)$. In Sec..., we show the conclusion and future work. More details about the algorithms and different variations of the algorithm are shown in the appendix.