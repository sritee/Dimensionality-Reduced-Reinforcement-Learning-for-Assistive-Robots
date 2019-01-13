# Dimensionality Reduced RL for Assistive Robots


This repository reproduces [1]'s results. This algorithm was proposed for increasing convergence speed of Reinforcement Learning involving high dimensional state spaces. We reproduce the paper's results,specifically the one on MountainCar3D environment -- a higher dimensional version of the original MountainCar. 

PCA Models folder contain the Projections which have been learned from demonstration data from the environment. (We use ~2500 samples, also in the same folder). We add a RobustScaler from scikit learn after the PCA normalization, which aids the learning process as TileCoding requires normalized states. The states are approximately normalized to 0 to 1 and fed to the Tile Coding algorithm, coded by Richard Sutton.


# References

[1] Curran, W., Brys, T., Aha, D., Taylor, M., & Smart, W. D. (2016, September). Dimensionality reduced reinforcement learning for assistive robots. In Proc. of Artificial Intelligence for Human-Robot Interaction at AAAI Fall Symposium Series.

[2] Richard Sutton, Tile Coding Software -- Reference Manual, Version 3 beta - http://incompleteideas.net/tiles/tiles3.html
