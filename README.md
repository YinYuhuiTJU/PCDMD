# PCDMD

**PCDMD** mploys a data-driven model using DMD, then calculates the residual of the physical equations, and finally corrects the predicted results using Kalman filter and gain coefficients. In this way, the PCDMD method can integrate the physics-informed equations with the data-driven model generated by DMD. The paper is published on [sample](https://sample)

In the following example, the animation is showing the goundtruth, DMD and PCDMD results, respectively from top to bottom. 

Diffusion
--------------
I used the Finite Differential Method to simulate the ground truth. Then I used the DMD and PCDMD to reconstruct and predict.

![calculated result1](./animation/Groundtruth.mp4)
![calculated result2](./animation/DMD.mp4)
![calculated result3](./animation/PCDMD.mp4)

Incompressible fluid solver
--------------
I used the [finite volume method (FVM)](https://en.wikipedia.org/wiki/Finite_volume_method) to represent the Navier-Stokes equation. The [Semi-Implicit Method for Pressure Linked Equations (SIMPLE)](https://en.wikipedia.org/wiki/SIMPLE_algorithm) was used to solve the velocity and pressure field iteratively.

![calculated result1](./animation/Groundtruth.mp4)
![calculated result2](./animation/DMD.mp4)
![calculated result3](./animation/PCDMD.mp4)

Navigating the code
--------------
The main loop of the provided code is simply a implementation of Finite Differential Method and PCDMD framework. 

*Under construction...* 

- Much more ...
