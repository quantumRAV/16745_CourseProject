/////////////////////////////////
//                             //
//       Buccal Mass Robot     //
//      Pressure Controller    //
//      PressurePort Class	   //
//                             //
// Created: Michael Bennington //
//          Ravesh Sukhnandan  //
// Last Updated: 1/15/2022 RS  //
//                             //
/////////////////////////////////

#include "Arduino.h"
#include "Wire.h"
#include "PressurePort.h"
#include "SPI.h"


/*! PressurePort
 *  @brief  Instantiates a new PressurePort Class
 *  @param  NumValveInflation
 *          Pin number controlling inflation valve (from supply pressure)
 *  @param  NumValveVacuum
 *          Pin number controlling vacuum valve (to ambient)
 *  @param  pressureSensor_SS
 *          Pin number controller the slave select for the Pressure Sensor (ELVH)
 *  @param  state
 *          0 = HOLD   : Don't open any valves, maintain current pressure
 *			1 = INFLATE: Open inflation valve and close vacuum valve. This will increase pressure in the Artificial Muscle.
 *			2 = VACUUM : Open vacuum valve and close inflation valve.  This will decrease pressure in the Artificial Muscle.
 *			3 = START  : Class has just been initialized.  
 *  @param  pressure_tolerance
 *          +- setpoint value which is considered good enough, after which the regulator will turn into the HOLD state and close inflation and deflation valves
 *  @param  min_pressure
 *          Minimum bound for the setpoint (psi)
 *  @param  max_pressure
 *          Maximum bound for the setpoint (psi)
 *  @param  Muscle
 *          Name of the muscle that this port is controlling
 */
PressurePort::PressurePort(int NumValveInflation, int NumValveVacuum, int pressureSensor_SS, PressureState state, float pressure_tolerance, float min_pressure, float max_pressure, String Muscle, float alpha )
{
	this->NumValveInflation = NumValveInflation;
	this->NumValveVacuum = NumValveVacuum;
	this->state = state;
	this->pressure_tolerance = pressure_tolerance;
	this->min_pressure = min_pressure;
	this->max_pressure = max_pressure;
	this->Muscle = Muscle;
	this->pressureSensor_SS = pressureSensor_SS;
	this->alpha = alpha;
	
	pinMode(this->NumValveInflation, OUTPUT);
	pinMode(this->NumValveVacuum, OUTPUT);
	pinMode(this->pressureSensor_SS, OUTPUT);
	
	digitalWrite(this->pressureSensor_SS, HIGH); //Deselect slave in the beginning
	
	if (this->state != START)
	{
		setValves(this-> state);
	}

}


PressurePort::PressurePort()//constructor that takes no arguments.  values are set to defaults.
{
}


PressurePort::~PressurePort()
{
}


/*! modulatePressure
 *  @brief  Modulates pressure to the setpoint by opening and closing the inflation and vacuum valves depending on the current pressure reading
 *  @param  newSetpoint
 *          Target pressure the controller will attempt to reach (psi)
 */
void PressurePort::modulatePressure(float newSetpoint)
{
	this->pressure = readPressure(true); 
	
	newSetpoint = min(max(newSetpoint, this->min_pressure),this->max_pressure); //check commanded setpoint to make sure it is within bounds
	this->setpoint = newSetpoint; 
	float pressure_diff = (this->pressure)-(this->setpoint);
	
	if (abs(pressure_diff)<=pressure_tolerance)
	{
		setValves(HOLD);
	}
	
	else if (pressure_diff<0) //means that actual pressure is less than commanded.  Need to inflate
	{
		setValves(INFLATE);
	}
	else if (pressure_diff>0) //means that actual pressure is less than commanded.  Need to vacuum
	{
		setValves(VACUUM);
	}
}




/*! modulatePressure_Activation
 *  @brief  Converts Activation value (0 to 1) to a pressure setpoint in psi.  It calls modulatePressure after converting the activation.
 *  @param  activation
 *          Neural activation value (0 to 1)
 */
void PressurePort::modulatePressure_Activation(float activation)
{
	float setpoint = ActivationToSetpoint(activation);
	modulatePressure(setpoint);
}





/*! modulatePressure_Activation_STOP
 *  @brief  Converts Activation value (0 to 1) to a pressure setpoint in psi.  It calls modulatePressure after converting the activation, and stops if you've reached the setpoint if Modulate_And_Stop is set to true
 *  @param  activation
 *          Neural activation value (0 to 1)
 *  @param 	Modulate_And_Stop: Set to true to stop modulating pressure as soon as you reach the setpoint within tolerance.  Set to false to continuously modulate.  
 */
void PressurePort::modulatePressure_Activation_STOP(float activation, byte bitArray[], byte MaxNumShiftRegisters, byte ArrayElementSize, bool Modulate_And_Stop )
{
	float newSetpoint = ActivationToSetpoint(activation);
	this->pressure = readPressure(true); 
	
	newSetpoint = min(max(newSetpoint, this->min_pressure),this->max_pressure); //check commanded setpoint to make sure it is within bounds

	if (this->reachedSetpoint == true and this->setpoint == newSetpoint and Modulate_And_Stop == true) //just hold the current pressure if you've reached the setpoint within tolerance and there hasn't been a change in the commanded setpoint and the functionality is set to Modulate and stop
	{
		setValve_state(HOLD,bitArray,MaxNumShiftRegisters, ArrayElementSize);
		return;

	}

	this->setpoint = newSetpoint; 
	float pressure_diff = (this->pressure)-(this->setpoint);
	if (abs(pressure_diff)<=pressure_tolerance)
	{
		setValve_state(HOLD,bitArray,MaxNumShiftRegisters, ArrayElementSize);
		this->reachedSetpoint = true;
	}
	
	else if (pressure_diff<0) //means that actual pressure is less than commanded.  Need to inflate
	{
		setValve_state(INFLATE,bitArray,MaxNumShiftRegisters, ArrayElementSize);
		this->reachedSetpoint = false;
	}
	else if (pressure_diff>0) //means that actual pressure is less than commanded.  Need to vacuum
	{
		setValve_state(VACUUM,bitArray,MaxNumShiftRegisters, ArrayElementSize);
		this->reachedSetpoint= false;
	}
}

/*! modulatePressure_Activation
 *  @brief  Converts Activation value (0 to 1) to a pressure setpoint in psi.  It calls modulatePressure after converting the activation.
 *  @param  activation
 *          Neural activation value (0 to 1)
 */
void PressurePort::modulatePressure_Activation(float activation, byte bitArray[], byte MaxNumShiftRegisters, byte ArrayElementSize  )
{
	modulatePressure_Activation_STOP(activation,bitArray,MaxNumShiftRegisters,ArrayElementSize,false); //don't run the modulate and stop function.
}




/*! setValves
 *  @brief  Opens and closes valves depending on the commanded state
 *  @param  state
 *          The state is described by the following.  Note that the Koganei valves are normally closed and the solenoid driver is non-inverting.
 *			HOLD: Keep both inflation and vacuum valves CLOSED.  Inflation valve pin: LOW.  Vacuum valve pin: LOW
 *			INFLATE: Keep inflation valve OPEN and vacuum valve CLOSED to allow muscle to inflate.  Inflation valve pin: HIGH.  Vacuum valve pin: LOW
 *			VACUUM: Keep inflation valve CLOSED and vacuum valve OPEN to allow muscle to deflate.  Inflation valve pin: LOW.  Vacuum valve pin: HIGH
 */
void PressurePort::setValves(PressureState state) 
{
	
	setValve_state(state,this->ShiftArray, this->MaxNumShiftRegisters, this->ArrayElementSize);
	//Serial.println(this->ShiftArray[3],BIN);
	WriteToShiftRegister(this->RCK, this->G_ENABLE, this->ShiftArray, this->MaxNumShiftRegisters);
	this->state = state;
	//Serial.println(ShiftArray);
	
	
}

/*! setValve_state
 *  @brief  Create bit stream for opening and closing valves.
 *  @param  state
 *          The state is described by the following.  Note that the Koganei valves are normally closed and the solenoid driver is non-inverting.
 *			HOLD: Keep both inflation and vacuum valves CLOSED.  Inflation valve pin: LOW.  Vacuum valve pin: LOW
 *			INFLATE: Keep inflation valve OPEN and vacuum valve CLOSED to allow muscle to inflate.  Inflation valve pin: HIGH.  Vacuum valve pin: LOW
 *			VACUUM: Keep inflation valve CLOSED and vacuum valve OPEN to allow muscle to deflate.  Inflation valve pin: LOW.  Vacuum valve pin: HIGH
 *			bitArray:  byte array to hold whether to open or close the valve
 *			MaxNumShiftRegisters: number of elements in bit Array
 *			ArrayElementSize: size of each element of the bit array. For bytes it should be 8 (bits).
 */
void PressurePort::setValve_state(PressureState state,byte bitArray[],byte MaxNumShiftRegisters, byte ArrayElementSize) 
{
	
	
	switch (state)
	{
		case HOLD:
			constructBitTrain(bitArray,(byte) NumValveInflation, LOW , MaxNumShiftRegisters, ArrayElementSize);
			constructBitTrain(bitArray,(byte)NumValveVacuum, LOW , MaxNumShiftRegisters, ArrayElementSize);
			break;
		
		case INFLATE:
			constructBitTrain(bitArray,(byte) NumValveInflation, HIGH , MaxNumShiftRegisters, ArrayElementSize);
			constructBitTrain(bitArray,(byte) NumValveVacuum, LOW , MaxNumShiftRegisters, ArrayElementSize);
			break;
			
		case VACUUM:
			constructBitTrain(bitArray,(byte) NumValveInflation, LOW , MaxNumShiftRegisters, ArrayElementSize);
			constructBitTrain(bitArray,(byte) NumValveVacuum, HIGH , MaxNumShiftRegisters, ArrayElementSize);
			break;
			
		case START:
			constructBitTrain(bitArray,(byte) NumValveInflation, LOW , MaxNumShiftRegisters, ArrayElementSize);
			constructBitTrain(bitArray,(byte) NumValveVacuum, LOW , MaxNumShiftRegisters, ArrayElementSize);
			break;
			
		case OPEN:
			constructBitTrain(bitArray,(byte) NumValveInflation, HIGH , MaxNumShiftRegisters, ArrayElementSize);
			constructBitTrain(bitArray,(byte) NumValveVacuum, HIGH , MaxNumShiftRegisters, ArrayElementSize);
		
	}

	
	
}



/*! ActivationToSetpoint
 *  @brief  Converts neural activation value to pressure setpoint (psi)
 *  @param  activation
 *          Neural Activation (0 to 1)
 *	@return	pressure setpoint (in psi)
 */
float PressurePort::ActivationToSetpoint(float activation)
{
	return(this->max_pressure*activation);
}

/*! readPressure
 *  @brief  Converts neural activation value to pressure setpoint (psi).  It will attempt to do an equally weighted moving average of the last n readings.  The value for n is set in the max_samples class member.
 *  @param  update_state (optional)
 *          Boolean whether to update member variable pressure.  By default this is true.
 *	@return	pressure reading from sensor (in psi).
 */
float PressurePort::readPressure(bool update_state)
{
	float pressure;
	float temperature;
	
	if (update_state == true)
	{
		readPressure_and_Temperature(pressure, this->temperature); //get raw pressure readings.  Perform moving average first before updating member variable
		//Serial.println(pressure);
		/* this->pressure_arr[this->current_iteration] = pressure; //place the pressure value into the array
		this->pressure = MovingAverage(this->pressure_arr,this->max_samples); //perform Moving Average2		this->current_iteration = (this->current_iteration + 1)%this->max_samples; //update the iteration */
		
		//Serial.println(this->pressure);
		/* float alpha = 1; //alpha = 1 works ok for lower pressures ~1.5 psi.  alpha = 0.001 works ok for larger pressures
		if (this->setpoint <= 4.5)
		{
			alpha = 1;
		}
		
		else
		{ 
			alpha = 0.1;
		}
		*/
		
		//float alpha = 0.07;
		
		this->pressure = pressure*(this->alpha) + (1-(this->alpha))*this->pressure; //exponentially weighted moving average  
		//this->pressure = pressure; //don't do moving average
		
		this->pressure = max(this->pressure,0);
		return(this->pressure);
		
		
	}
	else
	{
		readPressure_and_Temperature(pressure, temperature); //returns raw pressure value without filtering
		
		return(pressure);
	}
   
}

/*! readPressure_and_Temperature
 *  @brief  Converts neural activation value to pressure setpoint (psi).  It will attempt to do an equally weighted moving average of the last n readings.  The value for n is set in the max_samples class member.
 *  @param  pressure (by reference)
 *          Pass-by-reference variable which will return the pressure value in psi.
 *  @param  temperature (by reference)
 *          Pass-by-reference variable which will return the temperature value in degrees Celsius.
 */
void PressurePort::readPressure_and_Temperature(float & pressure, float & temperature)
{
   unsigned long receivedVal = 0x00000000;
   unsigned long receivedValP=0x00000000;
   double receivedValT;
   double calcT = 0;
   
   
   SPI.beginTransaction(SPISettings(500000, MSBFIRST, SPI_MODE0)); //begin SPI transaction
   digitalWrite(this->pressureSensor_SS, LOW); //Select slave

/* SPI.transfer(& receivedVal, 2);  //transfer 4 bytes
   digitalWrite(this->pressureSensor_SS, HIGH); //Deselect slave
   SPI.endTransaction();

   Serial.println(receivedVal, BIN) ;
   receivedValP = (receivedVal); //get pressure bits
   Seri
   al.println(receivedValP, BIN);
   pressure = this->max_pressure*receivedValP/13108 - (this->max_pressure*1638/13108); //calculate pressure in PSI */
   
   //SPI.transfer(& receivedVal, 4);  //transfer 4 bytes
   unsigned int long1 =0x00000000;
   unsigned int long2=0x00000000;
   unsigned int long3 = 0x00000000;
   unsigned int long4 = 0x00000000;
   
   long1 = SPI.transfer(0xFF);
   long2 = SPI.transfer(0xFF);
   //long3 = SPI.transfer(0xFF);
   //long4 = SPI.transfer(0xFF);
   
   
   digitalWrite(this->pressureSensor_SS, HIGH); //Deselect slave
   SPI.endTransaction();
   
   
   /* Serial.println("Start");
   Serial.println(long1, BIN);
   Serial.println(long2, BIN);
   Serial.println(long3, BIN);
   Serial.println(long4, BIN); */
   
   receivedValP = ((long1<<8)|long2)&PRESSURE;
   /* Serial.println(receivedVal, BIN);
   Serial.println("End"); */

	//Serial.println(receivedVal, BIN) ;
   //receivedValP = (receivedVal >>15) & PRESSURE; //get pressure bits
   /* Serial.println(receivedValP, BIN); */
   receivedValT = (receivedVal >>5) & TEMPERATURE; //get temperature bits
   
   temperature=receivedValT*200/(pow(2,11)-1) - 50; //calculate temperature in Celsius
   pressure = this->PressSensorMax*receivedValP/13108 - (this->PressSensorMax*1638/13108); //calculate pressure in PSI
   
}

/*! MovingAverage
 *  @brief  Equally weighted moving average of the last n readings.  The value for n is set in the max_samples class member.
 *  @param  pressure_arr 
 *          Array containing pressure values
 *  @param  size
 *          Length of pressure_arr
 */
float PressurePort::MovingAverage(float pressure_arr[], int size)
{
	float pressure = 0;
	for (int i = 0; i<size;i++)
	{
		pressure = pressure+pressure_arr[i];
		
		
		/* char pressI[10];
		dtostrf(pressure_arr[i], 3, 2, pressI);
		
		char buffer[40];
		sprintf(buffer, "Pressure %i %s",i, pressI);
		Serial.println(buffer); */
	}
	
	
	
	
	
	return(pressure/size);
	
	
	
}


void PressurePort::constructBitTrain(byte BitTrainArray[],byte position, bool state, byte MaxNumShiftRegisters, byte ArrayElementSize)
{
	byte mask = 1; //initialize mask
	
	byte arrayNum = MaxNumShiftRegisters - floor((position-1)/ArrayElementSize)-1;
	byte actualPos = (position-1)-ArrayElementSize*(floor((position-1)/ArrayElementSize)); //assuming position is one indexed
	
	mask = mask << actualPos; //shift into position
	/* Serial.print("Construct Bit TRAIN:");
	Serial.println(mask,BIN);
	Serial.println(actualPos); */
	

	if (state == HIGH)
	{
		BitTrainArray[arrayNum] = BitTrainArray[arrayNum]| mask;
		
	}
	
	else
	{
		BitTrainArray[arrayNum] = BitTrainArray[arrayNum] & (~mask);  //if that pin should be low
	}
	


	
	//Serial.println(BitTrainArray[arrayNum]);
	
	
	
}

void PressurePort::WriteToShiftRegister(int RCK, int G_ENABLE, byte arr [], int arr_size)
{ 

  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
  
  digitalWrite(RCK,LOW);

  for (int i = 0; i<arr_size;i++)
  {
	  SPI.transfer(arr[i]);
	  //Serial.println(arr[i]);

  }
  
  
  digitalWrite(RCK,HIGH); //copies data from input buffer to output buffer
  digitalWrite(G_ENABLE, LOW); //enables the chip when pulled low
  SPI.endTransaction();
}

/*! printState
 *  @brief  Prints out the current pressure reading, setpoint, minimum pressure, maximum pressure and pressure tolerance.
 */
void PressurePort::printState()
{
	char str[128];
	char pressure[10];
	char setpoint[10];
	char min_pressure[10];
	char max_pressure[10];
	char pressure_tolerance[10];
	
	
	dtostrf(this->pressure, 3, 2, pressure);
	dtostrf(this->setpoint, 3, 2, setpoint);
	dtostrf(this->min_pressure, 3, 2, min_pressure);
	dtostrf(this->max_pressure, 3, 2, max_pressure);
	dtostrf(this->pressure_tolerance, 3, 2, pressure_tolerance);
		
	snprintf(str,128,"Muscle:%s\tPressure:%s\tSetpoint:%s\tState:%i\tMinP:%s\tMaxP:%s\tTol:%s", this->Muscle.c_str(),pressure,setpoint,this->state,min_pressure,max_pressure,pressure_tolerance);
	Serial.println(str);
}