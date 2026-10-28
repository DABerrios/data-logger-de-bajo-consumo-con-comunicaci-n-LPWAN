/*!
 * \file      board-config.h
 *
 * \brief     Board configuration - STM32L476RG Custom Board + Fanstel LR62E (crystal)
 *
 * Pin mapping matches the custom board CubeMX .ioc configuration:
 *   SPI1: PA5(SCK), PA6(MISO), PA7(MOSI), PC4(NSS-software)
 *   Radio: PC5(NRESET), PB1(BUSY), PB0(DIO1-EXTI), PB2(ANT_SW)
 *   No UART debug (USB-CDC used separately by application)
 */
#ifndef __BOARD_CONFIG_H__
#define __BOARD_CONFIG_H__

#ifdef __cplusplus
extern "C"
{
#endif

/*
 * Force LR62E crystal variant and SX1262 board type.
 * These were project-level defines on the Nucleo; placing them here
 * avoids needing to edit .cproject IDE settings.
 */
#ifndef LR62E_FANSTEL_CRYSTAL
#define LR62E_FANSTEL_CRYSTAL
#endif

#ifndef SX1262MBXCAS
#define SX1262MBXCAS
#endif

/*!
 * LR62E crystal version: no TCXO, wakeup time = 0
 */
#define BOARD_TCXO_WAKEUP_TIME                      0

/*!
 * Board MCU pins definitions
 * Mapped to STM32L476RG Custom Board + LR62E wiring
 */

/* SX1262 Reset - PC5 */
#define RADIO_RESET                                 PC_5

/* SPI1 pins */
#define RADIO_MOSI                                  PA_7
#define RADIO_MISO                                  PA_6
#define RADIO_SCLK                                  PA_5

/* SPI1 chip select (software NSS) - PC4 */
#define RADIO_NSS                                   PC_4

/* SX1262 BUSY - PB1 */
#define RADIO_BUSY                                  PB_1

/* SX1262 DIO1 interrupt - PB0 (EXTI rising edge) */
#define RADIO_DIO_1                                 PB_0

/* LR62E Pin1 ANT-SW: RF switch/FEM enable - PB2
 * MUST be driven HIGH during TX/RX operations */
#define RADIO_ANT_SWITCH_POWER                      PB_2

/* LR62E has no DeviceSel pin exposed */
#define RADIO_DEVICE_SEL                            NC

/* No LEDs on custom board */
#define LED_1                                       NC
#define LED_2                                       NC

/* Debug pins not used */
#define RADIO_DBG_PIN_TX                            NC
#define RADIO_DBG_PIN_RX                            NC

#define OSC_LSE_IN                                  PC_14
#define OSC_LSE_OUT                                 PC_15

#define OSC_HSE_IN                                  PH_0
#define OSC_HSE_OUT                                 PH_1

#define SWCLK                                       PA_14
#define SWDAT                                       PA_13

#define I2C_SCL                                     PB_8
#define I2C_SDA                                     PB_9

/* No UART debug on custom board */
#define UART_TX                                     NC
#define UART_RX                                     NC

#ifdef __cplusplus
}
#endif

#endif // __BOARD_CONFIG_H__
