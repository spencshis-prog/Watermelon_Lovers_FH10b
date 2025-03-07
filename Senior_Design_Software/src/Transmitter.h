// File **********Transmitter.h***********
// Solution to Lab9 (not to be shown to students)
// Programs to implement transmitter functionality   
// EE445L Spring 2021
//    Jonathan W. Valvano 4/4/21
// 2-bit input, positive logic switches, positive logic software
#ifndef __TRANSMITTER_H__
#define __TRANSMITTER_H__


#include <stdint.h>
#include "../inc/CortexM.h"
#include "../inc/SysTickInts.h"

void Transmitter_Init(void);
void SysTick_Init(uint32_t period);

void selectSignalOne(void);
void selectSignalTwo(void);

void noSignal(void);
#endif
