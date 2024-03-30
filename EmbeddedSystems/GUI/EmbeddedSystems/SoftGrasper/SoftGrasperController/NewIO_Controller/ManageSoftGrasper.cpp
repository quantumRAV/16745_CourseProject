#include "PressurePort.h"
#include <SPI.h>
#include <Bounce.h>


const int RCK = 14;
const int G_ENABLE = 15;


//#define PressureCycle
//#define debug
#define DeflateALL
//#define ReadSingle
//#define DebugSingle
//#define ReadMultiple


// ------------- Initial Pressure Controller ---------------- //

// Controller arguments:
// int NumValveInflation, int NumValveVacuum, int pressureSensor_SS, PressureState state, float pressure_tolerance, float min_pressure, float max_pressure, String Muscle, float alpha
const int numPressurePorts = 12;  
const int numActivePorts = 7;
PressurePort Pa[numPressurePorts] = { PressurePort(1, 2, 28, PressurePort::START, 0.1, 0, 30, "I3", 0.01), //actuator 0 h
                                      PressurePort(3, 4, 25, PressurePort::START, 0.4, 0, 30, "Not Used", 0.12),//actuator 1 h
                                      PressurePort(7, 8, 24, PressurePort::START, 0.4, 0, 30, "Not Used", 0.12),  //broken //actuator 2
                                      PressurePort(6, 5, 10, PressurePort::START, 0.1, 0, 30, "I1_1", 0.12), //actuator 3 h
                                      PressurePort(9, 10, 9, PressurePort::START, 0.1, 0, 30, "I1_2", 0.12), //actuator 4 h
                                      PressurePort(11, 12, 8, PressurePort::START, 0.4, 0, 30, "Not Used", 0.5), //actuator 5 h
                                      PressurePort(13, 14, 7, PressurePort::START, 0.1, 0, 30, "I1_3", 0.12), //actuator 6 h
                                      PressurePort(15, 16, 6, PressurePort::START, 0.05, 0, 30, "Not Used", 0.4), //actuator 7 l
                                      PressurePort(17, 18, 5, PressurePort::START, 0.05, 0, 30, "JAW1", 0.35), //actuator 8 l 
                                      PressurePort(19, 20, 4, PressurePort::START, 0.05, 0, 30, "JAW2", 0.35), //actuator 9 l 
                                      PressurePort(24, 23, 3, PressurePort::START, 0.05, 0, 30, "Not Used", 0.5), //broken //actuator 10 h
                                      PressurePort(22, 21, 2, PressurePort::START, 0.05, 0, 30, "JAW3", 0.35) }; //actuator 11 l


//--------------Loop Timing--------------- //

unsigned long startTime = 0;     //uS
unsigned long endTime = 0;       //uS
unsigned long previousTime = 0;  //ms
unsigned long LoopTime = 1000;   //us
unsigned long t_init = 0;        //time when the loop starts to begin calculating sinusoid

IntervalTimer PressureControlTimer;

int count = 0;  //how many times through pressure control loop

// -------------- Shift Register Array for writing to the pressure controller valves ---------- //
byte bitArray[4] = { 0 };
byte MaxNumShiftRegisters = 4;
byte ArrayElementSize = 8;

// -------------  Pressure values for the muscle -----------------// 
float CommandValues[7]={0,0,0,0,0,0,0};
float P_I3=0;           // pressure for the I3-like circumferential muscle
float P_Jaw1=1;         // pressure for Soft Jaw 1
float P_Jaw2=1;         // pressure for Soft Jaw 2
float P_Jaw3=1;         // pressure for Soft Jaw 3
float P_I1_1=0;         // pressure for I1-like muscle for tilt control 
float P_I1_2=0;         // pressure for I1-like muscle for tilt control 
float P_I1_3=0;         // pressure for I1-like muscle for tilt control 

long HoldOverride = 0;  // read over serial if should override the pressure modulation function
// int JawCommand1 = false;
// int JawCommand2 = false;
// int JawCommand3 = false;
byte HoldValues[4]={0,0,0,0};

//-------------  Pressure Value -------------//
// rxBuffer: https://forum.arduino.cc/t/proper-method-of-receiving-uart-packet-data-far-exceeding-rx-buffer-size/611982/8



struct Transmission_Data_Structure
{
    char start[3] = ">>";
    byte protocol = 0;
    byte sizePayload = 48;
    float data[numPressurePorts];
    char end[3] = "<<";


}; 
int size_TransmissionData = sizeof(struct Transmission_Data_Structure);

struct Command_Data_Structure CommandData; 
struct Transmission_Data_Structure TransmissionData;


//-------------  Reading from Python -------------//
//Message Structure Lengths
#define MSG_HDR_LEN 2
#define MSG_TRLR_LEN 2
#define MSG_PAYLOAD_LEN_MAX 36 

//state names
#define ST_Header 0
#define ST_Payload 1
#define ST_Trailer 2

const char HeaderStr[MSG_HDR_LEN] = {'>', '>'};
const char TrailerStr[MSG_TRLR_LEN] = {'<', '<'};

byte Payload[MSG_PAYLOAD_LEN_MAX];


enum PortAction_ENUM
{
    HOLD = 0,
    INFLATE = 1,
    VACUUM = 2,
    START = 3,
    OPEN = 4,
    INFLATE_AND_STOP = 5,
    INFLATE_AND_MODULATE = 6,
    IGNORE = 7,

};


struct PortActions
{
    int port = 0;
    PortAction_ENUM action = IGNORE;
    float pressure = 0.0;
};

struct PortActions PortA[numPressurePorts] = { {0, IGNORE, 0.0},
                                            {1, IGNORE, 0.0},
                                            {2, IGNORE, 0.0},
                                            {3, IGNORE, 0.0},
                                            {4, IGNORE, 0.0},
                                            {5, IGNORE, 0.0},
                                            {6, IGNORE, 0.0},
                                            {7, IGNORE, 0.0},
                                            {8, IGNORE, 0.0},
                                            {9, IGNORE, 0.0},
                                            {10, IGNORE, 0.0},
                                            {11, IGNORE, 0.0}}; 


//-------Function defs:
void runActivationPattern();
void read_Commands();
void ReadSerial();
void executePressurePortActions();
void TransmitPressureReadings(const Transmission_Data_Structure* tdata);

void setup() {
  // put your setup code here, to run once:
    Serial.begin(460800);  //apparently can communicate at 1 MBaud 
    SPI.begin();
    pinMode(RCK, OUTPUT);

    digitalWrite(G_ENABLE, HIGH);
    pinMode(G_ENABLE, OUTPUT);
    delay(1000);

    //pinMode(SEROUT,INPUT);

    digitalWrite(G_ENABLE, LOW);


    SPI.beginTransaction(SPISettings(115200, MSBFIRST, SPI_MODE0));

    digitalWrite(RCK, LOW);
    unsigned long long1 = 0x00000000;
    unsigned long long2 = 0x00000000;
    long1 = SPI.transfer(0b00000000);  //(0b00100000);//relay connected to Drain 5 on 2nd Shift Register
    long2 = SPI.transfer(0b00000000);  //(0b00000100);//relay connected to Drain 2 on 1st Shift Register
    SPI.transfer(0b00000000);







    digitalWrite(RCK, HIGH);
    SPI.endTransaction();

    digitalWrite(G_ENABLE, LOW);


    //delay(1000000);
    digitalWrite(G_ENABLE, HIGH);


    #ifdef DeflateALL
        for (int i = 0; i < numPressurePorts; i++)
        {

        Pa[i].setValve_state(PressurePort::VACUUM,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed
        
       
        }
    #endif
    PressurePort::WriteToShiftRegister(PressurePort::RCK, PressurePort::G_ENABLE, bitArray, MaxNumShiftRegisters); //executes the bitArray to the shift registers to actually open or close the valves 
    delay(8000);





    // -- Pressure Cycle -- //
    #ifdef PressureCycle

        for (int i = 0; i < numPressurePorts; i++) {


            //byte bitArray[4]={0};
            Pa[i].setValves(PressurePort::INFLATE);
            Pa[i].WriteToShiftRegister(Pa[i].RCK, Pa[i].G_ENABLE, Pa[i].ShiftArray, 4);
            delay(1000);

            Pa[i].setValves(PressurePort::VACUUM);
            Pa[i].WriteToShiftRegister(Pa[i].RCK, Pa[i].G_ENABLE, Pa[i].ShiftArray, 4);
            delay(1000);
        }
    #endif

     // -- Debug A single Pressure Port -- //
    #ifdef DebugSingle

        int i = 0;
        int muscleToTest = 0;

        while (1) {
            Pa[muscleToTest].modulatePressure(12);
            float rawPress = Pa[muscleToTest].readPressure(false);  //without moving average
            char buffer[80];
            char pressure_s[10];
            dtostrf(rawPress, 3, 4, pressure_s);
            Serial.println(pressure_s);
            delay(1);

        }



    #endif





    // -- Debug A single Pressure Port that stops after 20 seconds -- //
    #ifdef DebugSingleStop

        int i = 0;
        int muscleToTest = 0;

        while (i<20000) {
            Pa[muscleToTest].modulatePressure(12);
            float rawPress = Pa[muscleToTest].readPressure(false);  //with moving average
            char buffer[80];
            char pressure_s[10];
            dtostrf(rawPress, 3, 4, pressure_s);
            Serial.println(pressure_s);
            delay(1);
            i++;
        }
        while (1)
        {
            Pa[muscleToTest].modulatePressure(0);
        }


    #endif


    // -- Read from a single pressure port -- //
    #ifdef ReadSingle
        const int FRR_Test_Port = 8;  //use act 6 to read pressure
            while (true) {
                int muscleToTest = FRR_Test_Port;
                float rawPress = Pa[muscleToTest].readPressure(false);  //without moving average
                char buffer[80];
                char pressure_s[10];
                dtostrf(rawPress, 3, 4, pressure_s);
                Serial.println(pressure_s);
                delay(1);
            }
    #endif

    #ifdef ReadMultiple
        
        const int FRR_Test_Port [12] = {0,1,2,3,4,5,6,7,8,9,10,11};  //Bank of 4 low pressure
            while (true) {
                String outputs ="";
                for (int j=0;j<12;j++)
                {
                    int muscleToTest = FRR_Test_Port[j];
                    float rawPress = Pa[muscleToTest].readPressure(false);  //without moving average
                    char buffer[80];
                    char pressure_s[10];
                    dtostrf(rawPress, 3, 4, pressure_s);
                    dtostrf(rawPress, 3, 4, pressure_s);
                    outputs = outputs + "," + pressure_s ;
                }
                Serial.println(outputs);
                delay(1);
            }

    #endif


    #ifdef GOcommand
        //wait for Go command
        bool exit = false;
        
        while(exit!=true)
        {
            

            if (Serial.available()) 
            {
                delayMicroseconds(1000);

                String test = Serial.readStringUntil('\n');
                if (test == "GO")
                {
                    exit = true;
                }
                delayMicroseconds(100);
                Serial.println("RD"); // send ready command
            }

        }
        Serial.println("ReadyToGo");

    #endif

    Serial.println("Teensy Ready!");
    delay(10);


}

void loop()
{
    // //Serial.println(endTime-startTime);
    // long fval = 0;
    // while (Serial.available()<=4)
    // {

    //   continue;

    // }  

    // for (int i=0;i<4;i++)
    // {
    //   HoldValues[i]= Serial.read();
      
    // }
    // float val = *((float*)(HoldValues));
    // CommandValues[3] = val;
    // runActivationPattern();
    readSerial();
    executePressurePortActions();
    TransmitPressureReadings(&TransmissionData);

    
    //delayMicroseconds(1000);

}

void ReadSerial()
{ //inspired by: https://forum.arduino.cc/t/proper-method-of-receiving-uart-packet-data-far-exceeding-rx-buffer-size/611982/8


    byte cByte;

    static int nIdx = 0;
    static int numPressure = 0;
    static int PressVal_ind [numPressurePorts] = {0,0,0,0,0,0,0,0,0,0,0,0};

    static byte stateRx = ST_Header; 

    if(Serial.available()>0)
    {
        while(Serial.available())
        {
            cByte=Serial.read();

            switch( stateRx)
            {
                case ST_Header:

                    if(cByte == HeaderStr[nIdx])
                    {
                        nIdx++;
                        if(nIdx == MSG_HDR_LEN-1) //reached end of the handler length
                        {
                            stateRx = ST_Payload;
                            nIdx = 0; //reset idx to 0 for use in the ST_Payload
                        }
                    }

                    else
                    {
                        nIdx = 0;
                    }

                break;

                case ST_Payload: //check Payload

                    Payload[nIdx] = cByte;

                    

                    if (nIdx<6)
                    {

                        PortA[2*nIdx].action = (int) (Payload[nIdx] >>4);
                        if (PortA[2*nIdx].action == INFLATE_AND_MODULATE or PortA[2*nIdx].action == INFLATE_AND_STOP)
                        {
                            PressVal_ind[numPressure] = 2*nIdx; 
                            numPressure++; //increment how many pressure bytes to expect
                        }  


                        PortA[2*nIdx+1].action = (int) (Payload[nIdx] & 0b00001111 );
                        if (PortA[2*nIdx+1].action == INFLATE_AND_MODULATE or PortA[2*nIdx].action == INFLATE_AND_STOP)
                        {
                            PressVal_ind[numPressure] = 2*nIdx + 1; 
                            numPressure++; //increment how many pressure bytes to expect
                        }

                        
                    }

                    else if (nIdx>=6 and nIdx< (6+numPressure))
                    {

                        int pressportIdx = PressVal_ind[nIdx-6]; //get the pressure value
                        PortA[pressportIdx].pressure = 12*Payload[nIdx]/255; //convert to a pressure value


                    }

                    nIdx++;

                    if (nIdx == 6+numPressure) //if you get the number of bytes you expected, then move on to the next step
                    {

                        numPressure = 0;
                        nIdx = 0;
                        stateRx = ST_Trailer;
                    }

                    //update the pressure port actions.

                break;

                case ST_Trailer: //check to see if Trailer has been received properly

                    if (cByte == TrailerStr[nIdx])
                    {
                        nIdx++;
                        if( nIdx == MSG_TRLR_LEN-1)
                        {
                            stateRx = ST_Header;
                            nIdx = 0;
                        }
                    }

                    else
                    {
                        nIdx = 0;
                        stateRx = ST_Header;
                    }

                break;

            }

            //execute the pressure port actions in case you are stuck in the Serial.available()
            executePressurePortActions();

        }

    }

}


void executePressurePortActions()
{
    //get commanded pressure values
    startTime = micros();
    //read_Commands();
    // Serial.println("DL");
    // delayMicroseconds(100);
    //  for (int v =0;v<numActivePorts;v++)
    // {
    //     Serial.println(CommandValues[v]);
    // }

    //--------------Timing begin
    unsigned long currentTime = micros();
    

    if (count == 0)
    {
        t_init = currentTime;
        count++;
        //P1.setValves(PressurePort::INFLATE);
    }

    unsigned long t = currentTime - t_init;
    float T = 1000000;

    //-----------------------------

    
    String outputs = "";
    for (int i = 0; i < numPressurePorts; i++)
    {

        PressurePort::PressureState PAction = PressurePort::HOLD;

        switch (PortA[i].action)
        {
            case HOLD:
                Pa[i].setValve_state(PressurePort::HOLD,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed


            break;

            case INFLATE:
                Pa[i].setValve_state(PressurePort::INFLATE,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed

            break;

            case VACUUM:
                Pa[i].setValve_state(PressurePort::VACUUM,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed

            break;

            case START:
                Pa[i].setValve_state(PressurePort::START,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed

            break;

            case OPEN:
                Pa[i].setValve_state(PressurePort::OPEN,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed
            break;

            case INFLATE_AND_STOP:
            break;
            
            case INFLATE_AND_MODULATE:
                float activation = PortA[i].pressure/30;
                Pa[i].modulatePressure_Activation(activation, bitArray, MaxNumShiftRegisters, ArrayElementSize);  //writes to bitArray which valves to open or close
            break;

            case IGNORE:
            break;

            
            
        }

        float rawPress = Pa[i].readPressure(false);  //without moving average
        char buffer[80];
        char pressure_s[10];
        dtostrf(rawPress, 3, 4, pressure_s);
        dtostrf(rawPress, 3, 4, pressure_s);
        outputs = outputs + "," + pressure_s ;
        
       
    }

    char str[outputs.length() + 1] = {};
    strcpy(str, outputs.c_str()); //convert to c_string

    
    PressurePort::WriteToShiftRegister(PressurePort::RCK, PressurePort::G_ENABLE, bitArray, MaxNumShiftRegisters); //executes the bitArray to the shift registers to actually open or close the valves 
    endTime = micros();

    TransmitPayload(HeaderStr,1,strlen(str),str,TrailerStr);

}

void TransmitPressureReadings(const Transmission_Data_Structure* tdata)
{
    Serial.write((const char*)tdata, size_TransmissionData); //https://arduino.stackexchange.com/questions/72138/send-structure-through-serial
}


void TransmitPayload(char startChar[], byte protocolType,byte numBytes,  byte payload[], char endChar[] )
{

    unsigned char buffer[255];
    buffer[0]=startChar[0];
    buffer[1]=startChar[1];
    buffer[2]=protocolType;
    buffer[3] = numBytes;
    for (int i = 0;i<numBytes;i++)
    {
        buffer[4+i] = payload[i];
    }
    buffer[numBytes+4] = endChar[0];
    buffer[numBytes+5] = endchar[1];

    unsigned int len = 6+numBytes;


    Serial.write(buffer,len);

}