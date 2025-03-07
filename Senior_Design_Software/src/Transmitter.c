// File **********Transmitter.c***********
// Lab 5
// Programs to implement transmitter functionality   
// EE445L Spring 2021
//    Jonathan W. Valvano 4/4/21
// Hardware/software assigned to transmitter
//   UART0, possible source of input data (conflicts with TExaSdisplay)
//   PC7-PC4 Port_C_Init, possible source of input data
//   Timer0A periodic interrupt to generate input
//   FreqFifo linkage from Encoder to SendData
//   Timer1A periodic interrupt create a sequence of frequencies
//   SSI1 PD3 PD1 PD0 TLV5616 DAC output
//   SysTick ISR output to DAC

#include <stdint.h>
#include "../inc/tm4c123gh6pm.h"
#include "../inc/Timer0A.h"
#include "../inc/Timer1A.h"
#include "../inc/fifo.h"
#include "../inc/tlv5616.h"
    // write this
   
#include <stdint.h>
#include "../inc/tm4c123gh6pm.h"
#include "../inc/CortexM.h"
#include "../inc/SysTickInts.h"
//#include "UART.h"
#define WAVE_TABLE_SIZE 256       


#define CR   0x0D
#define LF   0x0A


// **************SysTick_Init*********************
// Initialize SysTick periodic interrupts
// Input: interrupt period
//        Units of period are 12.5ns (assuming 80 MHz clock)
//        Maximum is 2^24-1
//        Minimum is determined by length of ISR
// Output: none





int signalOne = 0;
int signalTwo = 0;
int freq_entered = 0;

static uint32_t WaveIndex = 0;     // Current index in the sine wave table
//static uint32_t Period = 0;        // SysTick period for note frequency


void DelayTransmission(int ms);
void SysTick_SetPeriod(uint32_t);

static const uint16_t SineWave[WAVE_TABLE_SIZE] = {
127, 130, 133, 136, 139, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173,
176, 179, 182, 184, 187, 190, 193, 195, 198, 200,                                                                              203, 205, 208, 210, 213, 215,
217, 219, 221, 224, 226, 228, 229, 231, 233, 235, 236, 238, 239, 241, 242, 244,
245, 246, 247, 248, 249, 250, 251, 251, 252, 253, 253, 254, 254, 254, 254, 254,
255, 254, 254, 254, 254, 254, 253, 253, 252, 251, 251, 250, 249, 248, 247, 246,
245, 244, 242, 241, 239, 238, 236, 235, 233, 231, 229, 228, 226, 224, 221, 219,
217, 215, 213, 210, 208, 205, 203, 200, 198, 195, 193, 190, 187, 184, 182, 179,
176, 173, 170, 167, 164, 161, 158, 155, 152, 149, 146, 143, 139, 136, 133, 130,
127, 124, 121, 118, 115, 111, 108, 105, 102, 99, 96, 93, 90, 87, 84, 81,
78, 75, 72, 70, 67, 64, 61, 59, 56, 54, 51, 49, 46, 44, 41, 39,
37, 35, 33, 30, 28, 26, 25, 23, 21, 19, 18, 16, 15, 13, 12, 10,
9, 8, 7, 6, 5, 4, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8,
9, 10, 12, 13, 15, 16, 18, 19, 21, 23, 25, 26, 28, 30, 33, 35,
37, 39, 41, 44, 46, 49, 51, 54, 56, 59, 61, 64, 67, 70, 72, 75,
78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 115, 118, 121, 124,};




void SysTick_Init(uint32_t period){
	long sr;
  sr = StartCritical();
  NVIC_ST_CTRL_R = 0;         // disable SysTick during setup
  NVIC_ST_RELOAD_R = period-1;// reload value
  NVIC_ST_CURRENT_R = 0;      // any write to current clears it
  NVIC_SYS_PRI3_R = (NVIC_SYS_PRI3_R&0x00FFFFFF)|0x40000000; // priority 2
                              // enable SysTick with core clock and interrupts
  NVIC_ST_CTRL_R = 0x07;
  EndCritical(sr);
}



void selectSignalOne(void){
	signalOne =  1;
	signalTwo = 0;
	
}


void selectSignalTwo(void){
	signalOne =  0;
	signalTwo = 1;
	
}

void noSignal(void){
	signalOne =  0;
	signalTwo = 0;
	
}




void SysTick_SetPeriod(uint32_t newPeriod) {
    NVIC_ST_CTRL_R = 0;
    NVIC_ST_RELOAD_R = newPeriod - 1;
    NVIC_ST_CURRENT_R = 0;
    NVIC_ST_CTRL_R = 0x07; 
}




void SysTick_Handler(void){
	
		if(((GPIO_PORTA_DATA_R & 0x80) == 0x80)){
			DelayTransmission(5000);
			UART_OutString("Enter input frequency (Hz):\r\n");
			freq_entered = UART_InUDec();  
			freq_entered = 80000000 / (freq_entered * 256);
			SysTick_SetPeriod(freq_entered);
			UART_OutChar(CR); UART_OutChar(LF);
			UART_OutString("Input Recived.");
			UART_OutChar(CR); UART_OutChar(LF);
		}
	
		if(((GPIO_PORTF_DATA_R & 0x10) == 0x0)){
			selectSignalOne();
		}else{
			noSignal();
		}

		
		if(signalOne == 1){
			DAC_Out(SineWave[WaveIndex]);
			WaveIndex = (WaveIndex + 1) &0xFF;
		}
}


void DelayTransmission(int ms){
	while(ms > 0) {
        ms--;
    }
}





