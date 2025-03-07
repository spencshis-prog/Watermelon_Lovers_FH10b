


#include "../inc/tm4c123gh6pm.h"

void Switch_Init(void){
	  SYSCTL_RCGCGPIO_R |= 0x10;      // Activate Port E
  while((SYSCTL_PRGPIO_R & 0x10) == 0){} // Wait until Port E is ready
  //GPIO_PORTE_DIR_R &= ~0x07;      // Set PE0, PE1 as inputs
  GPIO_PORTE_DEN_R |= 0x07;       // Enable digital functionality on PE0, PE1
		
  SYSCTL_RCGCGPIO_R |= 0x20;     // activate port F
  while((SYSCTL_PRGPIO_R&0x20) == 0){}  // wait until ready 
  GPIO_PORTF_DIR_R |= 0x0A;      // make PF1 and PF3 output for toggling
  GPIO_PORTF_DEN_R |= 0x1A;      // enable digital I/O on PF1, PF3
	GPIO_PORTF_LOCK_R = 0x4C4F434B;    
  GPIO_PORTF_CR_R |= 0x01;           
  GPIO_PORTF_DIR_R &= ~0x11;     //change direction for switches    

  GPIO_PORTF_PUR_R |= 0x11;          
  GPIO_PORTF_AFSEL_R &= ~0x11;       
  GPIO_PORTF_DEN_R |= 0x11;  
		
		
	SYSCTL_RCGCGPIO_R |= 0x04;     // activate port C
  while((SYSCTL_PRGPIO_R&0x04) == 0){}  // wait until ready 
  GPIO_PORTC_DIR_R |= 0xF0;      // make PC7, PC6, PC5, and PC4 output for toggling
  GPIO_PORTC_DEN_R |= 0xF0;      // enable digital I/O on PC1, PC3
		
	SYSCTL_RCGCGPIO_R |= 0x01;     // activate port A
  while((SYSCTL_PRGPIO_R&0x01) == 0){}  // wait until ready 
  //GPIO_PORTC_DIR_R |= 0x80;      // make PA7
  GPIO_PORTA_DEN_R |= 0x80;      // enable digital I/O on PA7

}

/*// use for debugging profile
#define PF1       (*((volatile uint32_t *)0x40025008))
#define PF2       (*((volatile uint32_t *)0x40025010))
#define PF3       (*((volatile uint32_t *)0x40025020))
// global variable visible in Watch window of debugger
// increments at least once per button press
volatile uint32_t FallingEdges = 0;
void EdgeCounterPortF_Init(void){                          
  SYSCTL_RCGCGPIO_R |= 0x00000020; // (a) activate clock for port F
  FallingEdges = 0;               // (b) initialize counter
  GPIO_PORTF_LOCK_R = 0x4C4F434B;   // 2) unlock GPIO Port F (for PF0)
  GPIO_PORTF_CR_R = 0x1F;           // allow changes to PF4-0
  GPIO_PORTF_DIR_R |=  0x0E;        // output on PF3,2,1 (LEDs)
  GPIO_PORTF_DIR_R &= ~0x11;        // (c) make PF4,0 inputs (built-in buttons)
  GPIO_PORTF_AFSEL_R &= ~0x1F;      //     disable alt funct on PF4-0
  GPIO_PORTF_DEN_R |= 0x1F;         //     enable digital I/O on PF4-0   
  GPIO_PORTF_PCTL_R &= ~0x000FFFFF; // configure PF4-0 as GPIO
  GPIO_PORTF_AMSEL_R = 0;           //     disable analog functionality on PF
  GPIO_PORTF_PUR_R |= 0x11;         //     enable weak pull-up on PF4, PF0
  
  // PF4 edge-sensitive setup
  GPIO_PORTF_IS_R &= ~0x10;         // (d) PF4 is edge-sensitive
  GPIO_PORTF_IBE_R &= ~0x10;        //     PF4 is not both edges
  GPIO_PORTF_IEV_R &= ~0x10;        //     PF4 falling edge event
  GPIO_PORTF_ICR_R = 0x10;          // (e) clear flag4
  GPIO_PORTF_IM_R |= 0x10;          // (f) arm interrupt on PF4
  
  // PF0 edge-sensitive setup
  GPIO_PORTF_IS_R &= ~0x01;         // (d) PF0 is edge-sensitive
  GPIO_PORTF_IBE_R &= ~0x01;        //     PF0 is not both edges
  GPIO_PORTF_IEV_R &= ~0x01;        //     PF0 falling edge event
  GPIO_PORTF_ICR_R = 0x01;          // (e) clear flag0
  GPIO_PORTF_IM_R |= 0x01;          // (f) arm interrupt on PF0

  // Set priority and enable interrupts for Port F in the NVIC
  NVIC_PRI7_R = (NVIC_PRI7_R & 0xFF00FFFF) | 0x00A00000; // (g) priority 5
  NVIC_EN0_R = 0x40000000;          // (h) enable interrupt 30 in NVIC
}

void GPIOPortF_Handler(void){
  GPIO_PORTF_ICR_R = 0x10;      // acknowledge flag4
  FallingEdges = FallingEdges + 1;
  PF1 ^= 0x02; // profile
	if((GPIO_PORTF_DATA_R & 0x01) == 0){
			
			
			//Delay(1500000);
			selectSignalOne();
			
			DisableInterrupts();
			NVIC_ST_CTRL_R &= ~0x01; 
			NVIC_ST_RELOAD_R = SIGNALONE - 1;
			NVIC_ST_CURRENT_R = 0;
			NVIC_ST_CTRL_R |= 0x01; 
			EnableInterrupts();
			
			while((GPIO_PORTF_DATA_R & 0x01) == 0){sampleMic = 1;}	
			
			sampleMic = 0;
			
			noSignal();
			//GPIO_PORTC_DATA_R = GPIO_PORTC_DATA_R^0x20;
		}
		
		if((GPIO_PORTF_DATA_R & 0x10) == 0x00){
				//GPIO_PORTC_DATA_R = GPIO_PORTC_DATA_R^0x40;
			
			//Delay(1500000);
			selectSignalTwo();
			
			DisableInterrupts();
			NVIC_ST_CTRL_R &= ~0x01; 
			NVIC_ST_RELOAD_R = SIGNALTWO - 1;
			NVIC_ST_CURRENT_R = 0;
			NVIC_ST_CTRL_R |= 0x01; 
			EnableInterrupts();
			while((GPIO_PORTF_DATA_R & 0x10) == 0x00){sampleMic = 1;}
			sampleMic = 0;
			noSignal();
			//GPIO_PORTC_DATA_R = GPIO_PORTC_DATA_R^0x40;
		}
}*/


