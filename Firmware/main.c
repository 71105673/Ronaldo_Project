#include <stdio.h>
#include <stdint.h>
#include "sleep.h"
//#include "xparameters.h"

#define   Macro_Write_Block(dest, bits, data, pos)   ((dest) = (((unsigned)dest) & ~(((unsigned)bits)<<(pos))) | (((unsigned)data)<<(pos)))
#define Macro_Extract_Area(dest, bits, pos)      ((((unsigned)dest)>>(pos)) & (bits))

typedef struct {
    uint32_t START;
    uint32_t DONE;
} SCCB_TypeDef;

typedef struct {
    uint32_t cen_data;
    uint32_t selected_grid;
} VGA_TypeDef;

typedef struct {
   uint32_t CSR;
   uint32_t TXD;
   uint32_t RXD;
} UART_TypeDef;

typedef struct {
   uint32_t selected_kick;
} BTN_TypeDef;

#define AXI_BASE    0x44A00000

#define SCCB_BASE   0x44A10000
#define UART_BASE   0x44A20000
#define VGA_BASE    0x44A30000
#define BTN_BASE    0x44A00000


#define SCCB        ((SCCB_TypeDef *)SCCB_BASE)
#define UART      ((UART_TypeDef *)UART_BASE)
#define VGA         ((VGA_TypeDef  *)VGA_BASE)
#define BTN         ((BTN_TypeDef *)BTN_BASE)

#define MODE_GRID 0x1
#define MODE_FACE 0x2
#define MODE_BTN  0x3

void delay_ms(uint32_t ms);
void Update_VGA_Register_Value(VGA_TypeDef* vga, BTN_TypeDef* btn, uint32_t* vga_grid, uint32_t* vga_face, uint32_t* btn_in);
void UART_Init(UART_TypeDef * uart);
void UART_SendData(UART_TypeDef * uart, uint32_t data);
uint8_t UART_ReceiveDone(UART_TypeDef * uart);
uint32_t UART_ReceiveData(UART_TypeDef * uart);
uint8_t UART_IsChangeMode(uint32_t data);
void UART_SendModeChange(UART_TypeDef * uart, uint32_t data, uint8_t* mode);
void UART_SendValueData(UART_TypeDef * uart, uint32_t* vga_grid, uint32_t* vga_face, uint32_t* vga_btn, uint8_t mode);

uint32_t uart_data = 0;
uint8_t mode = MODE_GRID;

uint32_t vga_grid = 0;
uint32_t vga_face = 0;
uint32_t btn = 0;

//uint32_t grid_data = 0x3;
//uint32_t face_data = 0x08864;
//uint32_t btn_data = 0x5;

int main()
{
      SCCB->START = 1;             // start
       delay_ms(1);     //
       SCCB->START = 0;             // end

       UART_Init(UART);

   while(1)
   {
      Update_VGA_Register_Value(VGA, BTN, &vga_grid, &vga_face, &btn);

      if(UART_ReceiveDone(UART)) uart_data = UART_ReceiveData(UART); // uart receive data

      if(UART_IsChangeMode(uart_data)) UART_SendModeChange(UART, uart_data, &mode); //if 0x7 -> change mode

      UART_SendValueData(UART, &vga_grid, &vga_face, &btn, mode); //send data


      usleep(100000);
   }

    return 0;
}

void delay_ms(uint32_t ms)
{
    volatile uint32_t count;
    for(uint32_t i = 0; i < ms; i++)
    {

        for(count = 0; count < 100000; count++);
    }
}

void Update_VGA_Register_Value(VGA_TypeDef* vga, BTN_TypeDef* btn, uint32_t* vga_grid, uint32_t* vga_face, uint32_t* btn_in)
{
   uint32_t temp_grid;
   uint32_t temp_face;
   uint32_t temp_btn;

   temp_grid = vga->selected_grid;
   temp_face = vga->cen_data;
   temp_btn = btn->selected_kick;

   uint32_t ori_face_data = temp_face&0x3ff;
   if(ori_face_data < 80) temp_face = temp_face&0xffc00;
   else temp_face = (temp_face&0xffc00) + (ori_face_data - 80);

   if( (*vga_grid) != temp_grid){
      *vga_grid = temp_grid;
   }

   if( (*vga_face) != temp_face){
      *vga_face = temp_face;
   }

//   if( 0 != temp_btn){
//      *btn_in = temp_btn;
//   }
   *btn_in = temp_btn;
}

void UART_Init(UART_TypeDef * uart)
{
   uart->CSR = 0x23; //grid
}

void UART_SendData(UART_TypeDef * uart, uint32_t data)
{
   uart->TXD = data;
}

uint8_t UART_ReceiveDone(UART_TypeDef * uart)
{
   if( (UART->CSR&(0x01<<4)) ) return 1;
   else return 0;
}

uint32_t UART_ReceiveData(UART_TypeDef * uart)
{
   return uart->RXD;
}

uint8_t UART_IsChangeMode(uint32_t data)
{
   if (Macro_Extract_Area(data, 0x7, 5) == 0x7) return 1;
   else return 0;
}

void UART_SendModeChange(UART_TypeDef * uart, uint32_t data, uint8_t* mode)
{
   switch(data&0x7){
      case MODE_GRID: Macro_Write_Block((uart->CSR), 0x7, 0x1, 5); *mode = MODE_GRID; break;
      case MODE_FACE: Macro_Write_Block((uart->CSR), 0x7, 0x2, 5); *mode = MODE_FACE; break;
      case MODE_BTN:  Macro_Write_Block((uart->CSR), 0x7, 0x4, 5);  *mode = MODE_BTN; break;
      default:   Macro_Write_Block((uart->CSR), 0x7, 0x0, 5);  *mode = 0; break;
   }
}

void UART_SendValueData(UART_TypeDef * uart, uint32_t* vga_grid, uint32_t* vga_face, uint32_t* vga_btn, uint8_t mode)
{
   uint32_t temp_data;

   switch(mode){
      case MODE_GRID: {
         while((uart->CSR & 0x1<<2));
         Macro_Write_Block(*vga_grid, 0x7, MODE_GRID, 5);
         UART_SendData(uart, *vga_grid);
         //uart->TXD = *vga_grid;
         break;
      }
//      case FACE: {
//         while(!(uart->CSR & (0x1<<3)));
//         for(int i = 3; i >= 0; i--){
//            while (uart->CSR & (0x1<<2));
//            temp_data = face_data & (0x1f<<(i*5));
//            UART_SendData(uart, temp_data>>(i*5));
//         }
//         break;
//      }
      case MODE_FACE: {
          for (int i = 3; i >= 0; i--) {
              temp_data = (*vga_face >> (i*5)) & 0x1f;
              Macro_Write_Block(temp_data, 0x7, MODE_FACE, 5);

              while (uart->CSR & (0x1<<2));
              UART_SendData(uart, temp_data);
              //uart->TXD = temp_data;

              while (!(uart->CSR & (0x1<<3))); // 4) empty flag
          }
          break;
      }
      case MODE_BTN : {
         while((uart->CSR & 0x1<<2));
         Macro_Write_Block(*vga_btn, 0x7, MODE_BTN, 5);
         UART_SendData(uart, *vga_btn);
         //uart->TXD = *vga_btn;
         break;
      }
   }
}

