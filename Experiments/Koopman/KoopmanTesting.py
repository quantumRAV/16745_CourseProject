import numpy
from enum import Enum

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

class variable_type(Enum):
    CONTROL = 1 #variable will be actively controlled
    STATE = 2 #variable will not be controlled

class sample_type(Enum):
    UNIFORM = 1  # sample values uniformly
    GAUSSIAN = 2  # sample values from a gaussian distribution


class var_params:

    def __init__(self, x_type = variable_type.STATE,
                 control_min_max=[0,20],
                 units='psi',
                 samp_type = sample_type.GAUSSIAN,
                 samp_parameters = {"mean":10, "stddev":3},
                 x_hold_time = 30,
                 total_run_time=300,
                 samp_freq_Hz = 20):

        self.var_type = x_type
        self.control_min_max = control_min_max
        self.units = units
        self.samp_type = samp_type
        self.samp_parameters = samp_parameters
        self.samp_hold_time = x_hold_time
        self.total_run_time = total_run_time
        self.samp_freq_Hz = samp_freq_Hz


    def generate_random_sequence(self):

        num_points = math.floor(self.total_run_time*self.samp_freq_Hz/self.x_hold_time) #number of points in the experiment

        match self.var_type:

            case variable_type.STATE:

            case variable_type.CONTROL:

                match self.samp_type:

                    case sample_type.UNIFORM:

                    case sample_type.GAUSSIAN:





class koopman:

    def __init__(self):



