import numpy

'''
Pseudocode:
- Prompt user to get the following information: 
    - Control inputs: x,y,z,g,j1,j2,j3
    - min and max for controls
    - position of object
    - Specify whether to sample the variable by a uniform or gaussian distribution. 
      Specify The mean and std.dev of the gaussian to vary x,y,z
    - Specify the sample hold time for each control input (T_h)
    - Specify length of time to collect data in minutes -> convert to seconds (T_total)
    - Generate random control trajectories, with length: T_Total/T_h
    - Run data collection for set length of time.
        - Every T_h seconds, send the next random step input. Log positions and pressures. Repeat until T_Total
'''
class koopman:

    def __init__(self):

