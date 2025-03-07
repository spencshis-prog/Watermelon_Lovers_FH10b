// Receiver.h
// Programs to implement receiver functionality   
// ECE445L Fall 2024
//    Jonathan W. Valvano 8/30/24

// ----------------------------------------------------------------------------
// Hardware/software assigned to reciever
//   Timer2A ADC0 samples sound
//   Fifo3 linkage from  ADC to Decoder
//   main-loop runs decoder
//   PE3 PF3, PF2, PF1, LEDs
//   PA2-PA7, SSI0, ST7735R
#ifndef __RECEIVER_H__
#define __RECEIVER_H__
#include <stdint.h>
// write this
// ***solution***
// Solution to Lab5 (not to be shown to students)

// Sample time is 12.5ns*period
void Receiver_Init(uint32_t period);
uint8_t Receiver_Decode(void);
void ADC_Init(void);
int32_t ADC_In(void);

#endif
