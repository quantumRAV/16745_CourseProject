%% Looking into Time-Delay Embedding for Modeling Grasper
%
% Random step testing has been performed as inspo'd by Haggerty Sci Rob
% Paper. Based on step inputs to the McKibben actuator, pressure at the
% jaws was monitored as a proxy for the force exerted by the grasper
% system.
%
% We are looking to leverage Koopman Operator Theory, particularly with
% Time-Delay Embedding and a Hankel DMD approach to yield some dynamic
% model for this soft, non-linear system, such that our optimal controls
% approaches can be leveraged from class such that the grasper may be
% commanded to impart a particular force trajectory.
%
% N. Zimmerer & R. Sukhnandan

disp('hello')

%% Import Test Data
