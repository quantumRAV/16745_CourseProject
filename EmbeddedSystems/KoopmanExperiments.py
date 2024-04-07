from GUI.EmbeddedSystems.IntegratedSystem import IntegratedSystem as IS
from Experiments.Koopman.KoopmanTesting import koopman as KT





import asyncio
import aioconsole
import numpy as np
from enum import Enum


class koopman_experiments(IS.IntegratedSystem):


    def __init__(self):

        print("Initializing")
        super().__init__()
        print("Initializing_finished")











if __name__ == '__main__':

    IS = koopman_experiments()
    # async def run_Program():
    #     IS = IntegratedSystem()
    #     await IS.HardwareInitialize()
    #
    #     print("Finished Calibration")
    #     L = await asyncio.gather(
    #         IS.Normal_Mode(),
    #         IS.Calibration(),
    #         IS.SNS_Mode(),
    #         IS.ReadCommandLine(),
    #         IS.Read_Move_Hardware(),
    #         IS.TouchObject(),
    #         IS.datalog(),
    #         #IS.PressureRadiusCalibration(),
    #         #IS.capture_video()
    #     )
    #     #runp = asyncio.create_task(HandleProgram())
    #     #await runp
    #
    #     print('Before forever sleep')
    #     while True:
    #         await asyncio.sleep(1)
    #
    #
    #
    # try:
    #     #web.run_app(app)
    #     asyncio.run(run_Program())
    # except KeyboardInterrupt:
    #     "Exiting Program.  Thank you."