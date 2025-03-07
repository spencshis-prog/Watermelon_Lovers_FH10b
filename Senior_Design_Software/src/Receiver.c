// Receiver.c
// Lab 5
// Programs to implement receiver functionality   
// ECE445L Fall 2024
//    Jonathan W. Valvano 8/30/24

// ----------------------------------------------------------------------------
// Hardware/software assigned to receiver
//   Timer2A ADC0 samples sound
//   Fifo3 linkage from  ADC to Decoder
//   main-loop runs decoder
//   PE3 PF3, PF2, PF1, LEDs
//   PA2-PA7, SSI0, ST7735R

#include <stdint.h>
#include "../inc/tm4c123gh6pm.h"
#include "../inc/dsp.h"
#include "../inc/fifo.h"
#include <stdio.h>

#define MAX_SIZE 100

int queue[MAX_SIZE];
// write this




volatile int delay;



void ADC_Init(void){
// write this
	//while((SYSCTL_PRADC_R&0x0001) != 0x0001){};    // good code, but not yet implemented in simulator
	
	delay = 2000000;
	SYSCTL_RCGCADC_R |= 0x0001;   // 7) activate ADC0              
  SYSCTL_RCGCGPIO_R |= 0x10;  // 1) activate clock for Port E
  while((SYSCTL_PRGPIO_R&0x10) != 0x10){};
  GPIO_PORTE_DIR_R &= ~0x20;  // 3.8) make PE5 input
  GPIO_PORTE_AFSEL_R |= 0x20; // 4.8) enable alternate function on PE5
  GPIO_PORTE_DEN_R &= ~0x20;  // 5.8) disable digital I/O on PE5
  GPIO_PORTE_AMSEL_R |= 0x20; // 6.8) enable analog functionality on PE5
		
	delay = SYSCTL_RCGCADC_R;       // extra time to stabilize
	delay = SYSCTL_RCGCADC_R;       // extra time to stabilize
	delay = SYSCTL_RCGCADC_R;       // extra time to stabilize
  delay = SYSCTL_RCGCADC_R;

  ADC0_PC_R &= ~0xF;              // 7) clear max sample rate field
  ADC0_PC_R |= 0x1;               //    configure for 125K samples/sec
  ADC0_SSPRI_R = 0x0123;          // 8) Sequencer 3 is highest priority
  ADC0_ACTSS_R &= ~0x0008;        // 9) disable sample sequencer 3
  ADC0_EMUX_R &= ~0xF000;         // 10) seq3 is software trigger
  ADC0_SSMUX3_R &= ~0x000F;       // 11) clear SS3 field
  ADC0_SSMUX3_R += 8;             //    set channel Ain8
  ADC0_SSCTL3_R = 0x0006;         // 12) no TS0 D0, yes IE0 END0
  ADC0_IM_R &= ~0x0008;           // 13) disable SS3 interrupts
  ADC0_ACTSS_R |= 0x0008;         // 14) enable sample sequencer 3
}


int32_t ADC_In(void){
	uint32_t result;
   ADC0_PSSI_R = 0x0008;            // 1) initiate SS3
  while((ADC0_RIS_R&0x08)==0){};   // 2) wait for conversion done
  result = ADC0_SSFIFO3_R&0xFFF;   // 3) read result
  ADC0_ISC_R = 0x0008;             // 4) acknowledge completion
	
  return result;
	
}


