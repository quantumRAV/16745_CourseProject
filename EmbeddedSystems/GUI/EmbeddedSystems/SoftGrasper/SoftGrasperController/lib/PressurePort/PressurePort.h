
//////////////////////////////////////////
//                             		   	//
//       Buccal Mass Robot     		   	//
//      Pressure Controller    		   	//
//      PressurePort Class Header	   	//
//                             		   	//
// Created: Michael Bennington 			//
//          Ravesh Sukhnandan  		   	//
// Last Updated: 1/11/2022 RS  			//
//                             			//
//////////////////////////////////////////
#ifndef PressurePort_h
#define PressurePort_h


#include "Arduino.h"
#include "Wire.h"
#include "SPI.h"



class PressurePort{
protected:
	static const unsigned long PRESSURE = 0x3FFF;  //mask to extract bits for pressure sensor
	static const unsigned long TEMPERATURE = 0x7FF; //mask to extract bits for temperature sensor
	
	float setpoint = 0;				//target pressure.  (PSI)
	bool reachedSetpoint = false;	// true if you've reached the setpoint, false if you haven't reached it.
	
	static const int ShiftRegisterLength = 8;  //8 outputs for shift register
	static const int MaxNumShiftRegisters = 4;  //accomodate up to 32 outputs total: 8*4
	static const int ArrayElementSize = 8; //should be the size of the type of ShiftArray, e.g. int = 8, unsigned int =16, unsigned long = 32 etc.
	static const unsigned int ShiftArraySize = ceil(ShiftRegisterLength*MaxNumShiftRegisters/ArrayElementSize); //use ints to store data, should be 8 bits each
	


public:

	/** Pressure State */
	typedef enum {
	HOLD = 0,
	INFLATE = 1,
	VACUUM = 2,
	START = 3,
	OPEN = 4
	
	} PressureState;
		
	int NumValveInflation = 1; 		//order to drive solenoid open so that Artificial Muscle can inflate
	int NumValveVacuum = 2;	   		//order to drive solenoid open so that Artificial Muscle can deflate
	int pressureSensor_SS = 1;       //pin for the slave select of the pressure sensor
	int current_iteration = 0;       //iteration to 
	static const int max_samples = 10;			// number of iterations for moving average of pressure readings
	static const int max_size = 20;             //max size of buffer for averaging
	
	static const int G_ENABLE = 15; //for shift register: to enable the shift register
	static const int RCK = 14; //for shift register: to copy data from input buffer to output buffer
	byte ShiftArray[ShiftArraySize] = {0}; 
	
	PressureState state = START;	   	//Should the controller inflate, deflate or hold the current pressure?
	float pressure_tolerance = 0.25; 	//tolerance on target pressure.  (PSI)
	float min_pressure = 0;			//minimum pressure allowed, psi
	float max_pressure = 30;		//maximum pressure allowed, psi
	float PressSensorMin = 0;		//minimum pressure for pressure sensor psi
	float PressSensorMax = 30;		//maximum pressure for pressure sensor, psi
	float temperature = 0;			//current temperature, Celsius
	float pressure = 0; 			//current measured pressure.  (PSI)
	float pressure_arr[max_size]={0,0,0,0,0,0,0,0,0,0};			//array of 3 pressure values: n-2, n-1 and n
	float alpha = 1; 					//alpha for exponential moving average
	String Muscle;				//indicator for which muscle this pressure port is regulating 
	  
	
	PressurePort();

	/*
	int NumValveInflation:  pin for valve inflation
	int NumValveVacuum: pin for valve vacuum
	int pressureSensor_SS: pressure sensor select pin
	PressureState state: what state the pressure sensor is in
	float pressure_tolerance: tolerance of pressure to try to modulate within, in psi
	float min_pressure: minimum pressure for the pressure sensor (psi)
	float max_pressure: maximum pressure for the pressure sensor (psi)
	String Muscle: Name of the muscle
	float alpha:  alpha value for exponential moving average (pressure*(this->alpha) + (1-(this->alpha))*this->pressure)
	*/
	PressurePort(int NumValveInflation, int NumValveVacuum, int pressureSensor_SS, PressureState state, float pressure_tolerance, float min_pressure, float max_pressure, String Muscle, float alpha ); //constructor
	~PressurePort(); //destructor
	
	float readPressure(bool update_state = true); //returns the pressure reading in PSI.  By default, it will update PressurePort.pressure and PressurePort.temperature with this value.
	void readPressure_and_Temperature(float &pressure, float &temperature); //passes by reference the read temperature in Celsius and pressure in psi
	
	void modulatePressure(float newSetpoint); //function to modulate pressure.  By default, it will update PressurePort.pressure, PressurePort.setpoint and PressurePort.state
	void modulatePressure_Activation(float activation);
	float ActivationToSetpoint(float activation);
	void modulatePressure_Activation_STOP(float activation, byte bitArray[], byte MaxNumShiftRegisters, byte ArrayElementSize, bool Modulate_And_Stop );
	void modulatePressure_Activation(float activation, byte bitArray[], byte MaxNumShiftRegisters, byte ArrayElementSize  );
	
	void setValves(PressureState state);
	void setValve_state(PressureState state,byte bitArray[],byte MaxNumShiftRegisters, byte ArrayElementSize);
	void constructBitTrain(byte BitTrainArray[],byte position, bool state, byte MaxNumShiftRegisters, byte ArrayElementSize );
	static void WriteToShiftRegister(int RCK, int G_ENABLE, byte arr [], int arr_size);
	
	void printState();
	float MovingAverage(float pressure_arr[], int size);
	
	 
	
	
	
	

	
	
	
	

};



#endif