# quadruped-rl

Implementation of a data-based control architecture on the Solo-12 quadruped, using neural networks trained with reinforcement learning.

# Training

See [https://github.com/Gepetto/soloRL] for more details about the training environment.

# Dependencies

* Install a Python 3 version (for instance Python 3.8)

* Install PyBullet (simulation environment): `pip3 install --user pybullet`

* Install Inputs (to control the robot with a gamepad): `pip3 install --user inputs`

* Clone `git clone --recursive https://gitlab.laas.fr/paleziart/quadruped-rl.git`

* Compile the C++ part that runs the neural network:
    * `cd cpuMLP`
    * `mkdir build`
    * `cd build`
    * `cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/install -DPYTHON_EXECUTABLE=$(which python3) -DPYTHON_STANDARD_LAYOUT=ON`
    * `make`

* Run the controller in simulation: `python3 main_solo12_replay.py` 

# Deployment on a real Solo-12

* TODO

