/*!
 * \file      board.c
 *
 * \brief     Target board general functions implementation
 *
 * \copyright Revised BSD License, see section \ref LICENSE.
 *
 * \code
 *                ______                              _
 *               / _____)             _              | |
 *              ( (____  _____ ____ _| |_ _____  ____| |__
 *               \____ \| ___ |    (_   _) ___ |/ ___)  _ \
 *               _____) ) ____| | | || |_| ____( (___| | | |
 *              (______/|_____)_|_|_| \__)_____)\____)_| |_|
 *              (C)2013-2017 Semtech
 *
 * \endcode
 *
 * \author    Miguel Luis ( Semtech )
 *
 * \author    Gregory Cristian ( Semtech )
 */
#include "stm32l4xx.h"
#include "utilities.h"
#include "gpio.h"
// #include "adc.h"  // Not used on custom board (no HAL_ADC enabled)
#include "spi.h"
#include "i2c.h"
#include "uart.h"
#include "timer.h"
#include "sysIrqHandlers.h"
#include "board-config.h"
#include "lpm-board.h"
#include "rtc-board.h"

#if defined( SX1261MBXBAS ) || defined( SX1262MBXCAS ) || defined( SX1262MBXDAS )
    #include "sx126x-board.h"
#elif defined( LR1110MB1XXS )
    #include "lr1110-board.h"
#elif defined( SX1272MB2DAS)
    #include "sx1272-board.h"
#elif defined( SX1276MB1LAS ) || defined( SX1276MB1MAS )
    #include "sx1276-board.h"
#endif
#include "board.h"

/*!
 * Unique Devices IDs register set ( STM32L4xxx )
 */
#define         ID1                                 ( 0x1FFF7590 )
#define         ID2                                 ( 0x1FFF7594 )
#define         ID3                                 ( 0x1FFF7594 )

/*!
 * LED GPIO pins objects
 */
Gpio_t Led1;
Gpio_t Led2;

/*
 * MCU objects
 */
// Adc_t  Adc;  // Not used on custom board
Uart_t Uart2;

#if defined( LR1110MB1XXS )
    extern lr1110_t LR1110;
#endif

/*!
 * Initializes the unused GPIO to a know status
 */
static void BoardUnusedIoInit( void );

/*!
 * Initializes FLASH memory operations for EEPROM_Emul package
 */
static void InitFlashMemoryOperations( void );

/*!
 * System Clock Configuration
 */
static void SystemClockConfig( void );

/*!
 * System Clock Re-Configuration when waking up from STOP mode
 */
static void SystemClockReConfig( void );

/*!
 * Flag to indicate if the MCU is Initialized
 */
static bool McuInitialized = false;

/*!
 * Flag used to indicate if board is powered from the USB
 */
static bool UsbIsConnected = false;

/*!
 * UART2 FIFO buffers size
 */
#define UART2_FIFO_TX_SIZE                                1024
#define UART2_FIFO_RX_SIZE                                1024

uint8_t Uart2TxBuffer[UART2_FIFO_TX_SIZE];
uint8_t Uart2RxBuffer[UART2_FIFO_RX_SIZE];

void BoardCriticalSectionBegin( uint32_t *mask )
{
    *mask = __get_PRIMASK( );
    __disable_irq( );
}

void BoardCriticalSectionEnd( uint32_t *mask )
{
    __set_PRIMASK( *mask );
}

void BoardInitPeriph( void )
{

}

void BoardInitMcu( void )
{
    if( McuInitialized == false )
    {
        /* Custom board: HAL_Init and SystemClock are done by main.c before
         * BoardInitMcu is called.  Do NOT call them again here. */

        InitFlashMemoryOperations( );

        // LEDs (NC on custom board — harmless no-ops)
        GpioInit( &Led1, LED_1, PIN_OUTPUT, PIN_PUSH_PULL, PIN_NO_PULL, 0 );
        GpioInit( &Led2, LED_2, PIN_OUTPUT, PIN_PUSH_PULL, PIN_NO_PULL, 0 );

        /* UART disabled on custom board — no debug serial port.
         * USB-CDC is managed by the application (main.c), not by the Semtech stack. */

        RtcInit( );

        BoardUnusedIoInit( );
        if( GetBoardPowerSource( ) == BATTERY_POWER )
        {
            // Disables OFF mode - Enables lowest power mode (STOP)
            LpmSetOffMode( LPM_APPLI_ID, LPM_DISABLE );
        }
    }
    else
    {
        SystemClockReConfig( );
    }

    // AdcInit( &Adc, NC );  // Not used on custom board

#if defined( SX1261MBXBAS ) || defined( SX1262MBXCAS ) || defined( SX1262MBXDAS )
    SpiInit( &SX126x.Spi, SPI_1, RADIO_MOSI, RADIO_MISO, RADIO_SCLK, NC );
    SX126xIoInit( );
#elif defined( LR1110MB1XXS )
    SpiInit( &LR1110.spi, SPI_1, RADIO_MOSI, RADIO_MISO, RADIO_SCLK, NC );
    lr1110_board_init_io( &LR1110 );
#elif defined( SX1272MB2DAS )
    SpiInit( &SX1272.Spi, SPI_1, RADIO_MOSI, RADIO_MISO, RADIO_SCLK, NC );
    SX1272IoInit( );
#elif defined( SX1276MB1LAS ) || defined( SX1276MB1MAS )
    SpiInit( &SX1276.Spi, SPI_1, RADIO_MOSI, RADIO_MISO, RADIO_SCLK, NC );
    SX1276IoInit( );
#endif

    if( McuInitialized == false )
    {
        McuInitialized = true;
#if defined( SX1261MBXBAS ) || defined( SX1262MBXCAS ) || defined( SX1262MBXDAS )
        SX126xIoDbgInit( );
        // WARNING: If necessary the TCXO control is initialized by SX126xInit function.
#elif defined( LR1110MB1XXS )
        lr1110_board_init_dbg_io( &LR1110 );
        // WARNING: If necessary the TCXO control is initialized by SX126xInit function.
#elif defined( SX1272MB2DAS )
        SX1272IoDbgInit( );
        SX1272IoTcxoInit( );
#elif defined( SX1276MB1LAS ) || defined( SX1276MB1MAS )
        SX1276IoDbgInit( );
        SX1276IoTcxoInit( );
#endif
    }
}

void BoardResetMcu( void )
{
    CRITICAL_SECTION_BEGIN( );

    //Restart system
    NVIC_SystemReset( );
}

void BoardDeInitMcu( void )
{
    // AdcDeInit( &Adc );  // Not used on custom board

#if defined( SX1261MBXBAS ) || defined( SX1262MBXCAS ) || defined( SX1262MBXDAS )
    SpiDeInit( &SX126x.Spi );
    SX126xIoDeInit( );
#elif defined( LR1110MB1XXS )
    SpiDeInit( &LR1110.spi );
    lr1110_board_deinit_io( &LR1110 );
#elif defined( SX1272MB2DAS )
    SpiDeInit( &SX1272.Spi );
    SX1272IoDeInit( );
#elif defined( SX1276MB1LAS ) || defined( SX1276MB1MAS )
    SpiDeInit( &SX1276.Spi );
    SX1276IoDeInit( );
#endif
}

/*!
 * \brief Re-initialises the radio after V_TRANS power cycle.
 *
 * When the custom board cuts power to the LR62E (V_TRANS off in STOP2),
 * the SX1262 loses ALL register state.  This function performs a full
 * cold-start identical to what RadioInit() does at boot.
 */
void BoardRadioPowerCycleInit( void )
{
#if defined( SX1261MBXBAS ) || defined( SX1262MBXCAS ) || defined( SX1262MBXDAS )
    SpiInit( &SX126x.Spi, SPI_1, RADIO_MOSI, RADIO_MISO, RADIO_SCLK, NC );
    SX126xIoInit( );

    SX126xReset( );
    SX126xSetOperatingMode( MODE_STDBY_RC );
    SX126xWaitOnBusy( );

    SX126xIoRfSwitchInit( );
    SX126xAntSwOn( );

    extern void RadioOnDioIrq( void* context );
    SX126xIoIrqInit( RadioOnDioIrq );

    SX126xSetRegulatorMode( USE_DCDC );
    SX126xSetBufferBaseAddress( 0x00, 0x00 );
    SX126xSetDioIrqParams( IRQ_RADIO_ALL, IRQ_RADIO_ALL,
                           IRQ_RADIO_NONE, IRQ_RADIO_NONE );

    /* Full calibration after cold start */
    {
        uint8_t calAll = 0x7F;
        SX126xWriteCommand( RADIO_CALIBRATE, &calAll, 1 );
        SX126xWaitOnBusy( );
    }

    SX126xCalibrateImage( 915900000 );

    /* Add critical registers to retention list */
    {
        extern void RadioAddRegisterToRetentionList( uint16_t registerAddress );
        RadioAddRegisterToRetentionList( REG_RX_GAIN );
        RadioAddRegisterToRetentionList( REG_TX_MODULATION );
    }

    /* ---- Fix stale driver state after power cycle ----
     * The radio HW was reset but the driver's RAM variables still hold
     * values from before sleep.  Critical fixes:
     *  1) RadioPublicNetwork.Current = false → forces sync-word rewrite
     *  2) IrqFired = false → clears stale IRQ from prior session
     *  3) Timer re-init → clears stale linked-list pointers
     */
    {
        extern void RadioResetAfterPowerCycle( void );
        RadioResetAfterPowerCycle( );
    }

    /* ---- Re-enable RTC alarm IRQ for MAC timer system ----
     * RtcInit() skips re-init because RtcInitialized==true (RAM survived
     * STOP2).  But HAL_RTC_DeInit(&hrtc) before sleep cleared the alarm
     * registers.  Re-enable the NVIC line so TimerIrqHandler can fire
     * for TX timeout, RX window scheduling, etc.
     */
    {
        extern bool RtcInitialized;
        RtcInitialized = false;
        RtcInit( );
    }

    /* ---- Force MAC state to IDLE ----
     * After power-cycling the radio the MAC may be stuck in
     * TX_RUNNING from the previous cycle (we abort RX windows
     * with TimerStopAll).  LoRaMacStart() unconditionally sets
     * MacState = LORAMAC_IDLE so the next LmHandlerSend succeeds.
     */
    {
        typedef int LoRaMacStatus_t;          /* avoid full header include */
        extern LoRaMacStatus_t LoRaMacStart( void );
        LoRaMacStart( );
    }
#endif
}

uint32_t BoardGetRandomSeed( void )
{
    return ( ( *( uint32_t* )ID1 ) ^ ( *( uint32_t* )ID2 ) ^ ( *( uint32_t* )ID3 ) );
}

void BoardGetUniqueId( uint8_t *id )
{
    id[7] = ( ( *( uint32_t* )ID1 )+ ( *( uint32_t* )ID3 ) ) >> 24;
    id[6] = ( ( *( uint32_t* )ID1 )+ ( *( uint32_t* )ID3 ) ) >> 16;
    id[5] = ( ( *( uint32_t* )ID1 )+ ( *( uint32_t* )ID3 ) ) >> 8;
    id[4] = ( ( *( uint32_t* )ID1 )+ ( *( uint32_t* )ID3 ) );
    id[3] = ( ( *( uint32_t* )ID2 ) ) >> 24;
    id[2] = ( ( *( uint32_t* )ID2 ) ) >> 16;
    id[1] = ( ( *( uint32_t* )ID2 ) ) >> 8;
    id[0] = ( ( *( uint32_t* )ID2 ) );
}

/*!
 * Factory power supply
 */
#define VDDA_VREFINT_CAL ( ( uint32_t ) 3000 )  // mV

/*!
 * VREF calibration value
 */
#define VREFINT_CAL ( *( uint16_t* ) ( ( uint32_t ) 0x1FFF75AA ) )

/*
 * Internal temperature sensor, parameter TS_CAL1: TS ADC raw data acquired at
 * a temperature of 110 DegC (+-5 DegC), VDDA = 3.3 V (+-10 mV).
 */
#define TEMP30_CAL_ADDR ( *( uint16_t* ) ( ( uint32_t ) 0x1FFF75A8 ) )

/* Internal temperature sensor, parameter TS_CAL2: TS ADC raw data acquired at
 *a temperature of  30 DegC (+-5 DegC), VDDA = 3.3 V (+-10 mV). */
#define TEMP110_CAL_ADDR ( *( uint16_t* ) ( ( uint32_t ) 0x1FFF75CA ) )

/* Vdda value with which temperature sensor has been calibrated in production
   (+-10 mV). */
#define VDDA_TEMP_CAL ( ( uint32_t ) 3000 )

/*!
 * Battery thresholds
 */
#define BATTERY_MAX_LEVEL 3000       // mV
#define BATTERY_MIN_LEVEL 2400       // mV
#define BATTERY_SHUTDOWN_LEVEL 2300  // mV

#define BATTERY_LORAWAN_UNKNOWN_LEVEL 255
#define BATTERY_LORAWAN_MAX_LEVEL 254
#define BATTERY_LORAWAN_MIN_LEVEL 1
#define BATTERY_LORAWAN_EXT_PWR 0

#define COMPUTE_TEMPERATURE( TS_ADC_DATA, VDDA_APPLI )                                                          \
    ( ( ( ( ( ( ( int32_t )( ( TS_ADC_DATA * VDDA_APPLI ) / VDDA_TEMP_CAL ) - ( int32_t ) TEMP30_CAL_ADDR ) ) * \
            ( int32_t )( 110 - 30 ) )                                                                           \
          << 8 ) /                                                                                              \
        ( int32_t )( TEMP110_CAL_ADDR - TEMP30_CAL_ADDR ) ) +                                                   \
      ( 30 << 8 ) )

static uint16_t BatteryVoltage = BATTERY_MAX_LEVEL;

uint16_t BoardBatteryMeasureVoltage( void )
{
    // ADC not used on custom board — return nominal 3.3V
    return 3300;
}

uint32_t BoardGetBatteryVoltage( void )
{
    return BatteryVoltage;
}

uint8_t BoardGetBatteryLevel( void )
{
    uint8_t batteryLevel = 0;

    BatteryVoltage = BoardBatteryMeasureVoltage( );

    if( GetBoardPowerSource( ) == USB_POWER )
    {
        batteryLevel = BATTERY_LORAWAN_EXT_PWR;
    }
    else
    {
        if( BatteryVoltage >= BATTERY_MAX_LEVEL )
        {
            batteryLevel = BATTERY_LORAWAN_MAX_LEVEL;
        }
        else if( ( BatteryVoltage > BATTERY_MIN_LEVEL ) && ( BatteryVoltage < BATTERY_MAX_LEVEL ) )
        {
            batteryLevel =
                ( ( 253 * ( BatteryVoltage - BATTERY_MIN_LEVEL ) ) / ( BATTERY_MAX_LEVEL - BATTERY_MIN_LEVEL ) ) + 1;
        }
        else if( ( BatteryVoltage > BATTERY_SHUTDOWN_LEVEL ) && ( BatteryVoltage <= BATTERY_MIN_LEVEL ) )
        {
            batteryLevel = 1;
        }
        else  // if( BatteryVoltage <= BATTERY_SHUTDOWN_LEVEL )
        {
            batteryLevel = BATTERY_LORAWAN_UNKNOWN_LEVEL;
        }
    }
    return batteryLevel;
}

int16_t BoardGetTemperature( void )
{
    // ADC temp sensor not used on custom board (TMP117 used instead)
    return ( 25 << 8 );  // Return 25°C nominal
}

static void BoardUnusedIoInit( void )
{
    HAL_DBGMCU_EnableDBGSleepMode( );
    HAL_DBGMCU_EnableDBGStopMode( );
    HAL_DBGMCU_EnableDBGStandbyMode( );
}

void SystemClockConfig( void )
{
    RCC_OscInitTypeDef RCC_OscInitStruct = { 0 };
    RCC_ClkInitTypeDef RCC_ClkInitStruct = { 0 };
    RCC_PeriphCLKInitTypeDef PeriphClkInit = { 0 };

    __HAL_RCC_PWR_CLK_ENABLE( );

    __HAL_PWR_VOLTAGESCALING_CONFIG( PWR_REGULATOR_VOLTAGE_SCALE1 );

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI | RCC_OSCILLATORTYPE_LSE;
    RCC_OscInitStruct.MSIState            = RCC_MSI_ON;
    RCC_OscInitStruct.LSEState            = RCC_LSE_ON;
    RCC_OscInitStruct.MSIClockRange       = RCC_MSIRANGE_6;
    RCC_OscInitStruct.MSICalibrationValue = RCC_MSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState        = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource       = RCC_PLLSOURCE_MSI;
    RCC_OscInitStruct.PLL.PLLM            = 1;
    RCC_OscInitStruct.PLL.PLLN            = 40;
    RCC_OscInitStruct.PLL.PLLP            = RCC_PLLP_DIV7;
    RCC_OscInitStruct.PLL.PLLQ            = RCC_PLLQ_DIV4;
    RCC_OscInitStruct.PLL.PLLR            = RCC_PLLR_DIV2;
    if( HAL_RCC_OscConfig( &RCC_OscInitStruct ) != HAL_OK )
    {
        assert_param( LMN_STATUS_ERROR );
    }

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | 
                                  RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    if( HAL_RCC_ClockConfig( &RCC_ClkInitStruct, FLASH_LATENCY_4 ) != HAL_OK )
    {
        assert_param( LMN_STATUS_ERROR );
    }

    PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_RTC;
    PeriphClkInit.RTCClockSelection = RCC_RTCCLKSOURCE_LSE;
    if( HAL_RCCEx_PeriphCLKConfig( &PeriphClkInit ) != HAL_OK )
    {
        assert_param( LMN_STATUS_ERROR );
    }

    HAL_SYSTICK_Config( HAL_RCC_GetHCLKFreq( ) / 1000 );

    HAL_SYSTICK_CLKSourceConfig( SYSTICK_CLKSOURCE_HCLK );

    // SysTick_IRQn interrupt configuration
    HAL_NVIC_SetPriority( SysTick_IRQn, 0, 0 );
}

/*!
 * \brief  Programmable Voltage Detector (PVD) Configuration
 *         PVD set to level 6 for a threshold around 2.9V.
 * \param  None
 * \retval None
 */
static void PVD_Config( void )
{
    PWR_PVDTypeDef sConfigPVD;
    sConfigPVD.PVDLevel = PWR_PVDLEVEL_6;
    sConfigPVD.Mode     = PWR_PVD_MODE_IT_RISING;
    if( HAL_PWR_ConfigPVD( &sConfigPVD ) != HAL_OK )
    { 
        assert_param( LMN_STATUS_ERROR );
    }

    // Enable PVD
    HAL_PWR_EnablePVD( );

    // Enable and set PVD Interrupt priority
    HAL_NVIC_SetPriority( PVD_PVM_IRQn, 0, 0 );
    HAL_NVIC_EnableIRQ( PVD_PVM_IRQn );
}

/*!
 * \brief Initializes the EEPROM emulation module.
 *
 * \remark This function is defined in eeprom-board.c file
 */
void EepromMcuInit( void );

static void InitFlashMemoryOperations( void )
{
    // Enable and set FLASH Interrupt priority
    // FLASH interrupt is used for the purpose of pages clean up under interrupt
    HAL_NVIC_SetPriority( FLASH_IRQn, 0, 0 );
    HAL_NVIC_EnableIRQ( FLASH_IRQn );

    // Unlock the Flash Program Erase controller
    HAL_FLASH_Unlock( );

#if defined (STM32L4R5xx) || defined (STM32L4R7xx) || defined (STM32L4R9xx) || defined (STM32L4S5xx) || defined (STM32L4S7xx) || defined (STM32L4S9xx)
    // Clear OPTVERR bit and PEMPTY flag if set
    if( __HAL_FLASH_GET_FLAG( FLASH_FLAG_OPTVERR ) != RESET )
    {
        __HAL_FLASH_CLEAR_FLAG( FLASH_FLAG_OPTVERR );
    }

    if( __HAL_FLASH_GET_FLAG( FLASH_FLAG_PEMPTY ) != RESET )
    {
        __HAL_FLASH_CLEAR_FLAG( FLASH_FLAG_PEMPTY );
    }
#endif /* defined (STM32L4R5xx) || defined (STM32L4R7xx) || defined (STM32L4R9xx) || defined (STM32L4S5xx) || defined (STM32L4S7xx) || defined (STM32L4S9xx) */

    // Enable Power Control clock
    __HAL_RCC_PWR_CLK_ENABLE();
#if defined (USE_STM32L4XX_NUCLEO_144)
    HAL_PWR_DisableWakeUpPin( PWR_WAKEUP_PIN2 );
#endif /* defined (USE_STM32L4XX_NUCLEO_144) */

    // Configure Programmable Voltage Detector (PVD) (optional)
    // PVD interrupt is used to suspend the current application flow in case
    // a power-down is detected, allowing the flash interface to finish any
    // ongoing operation before a reset is triggered.
    PVD_Config( );

    // Initialize the EEPROM emulation driver
    EepromMcuInit( );

    // Lock the Flash Program Erase controller
    HAL_FLASH_Lock( );
}

void SystemClockReConfig( void )
{
      RCC_ClkInitTypeDef RCC_ClkInitStruct = { 0 };
      RCC_OscInitTypeDef RCC_OscInitStruct = { 0 };
      uint32_t pFLatency = 0;

    CRITICAL_SECTION_BEGIN( );

    // In case nvic had a pending IT, the arm doesn't enter stop mode
    // Hence the pll is not switched off and will cause HAL_RCC_OscConfig return 
    // an error
    if ( __HAL_RCC_GET_SYSCLK_SOURCE() != RCC_CFGR_SWS_PLL )
    {
        // Enable Power Control clock
        __HAL_RCC_PWR_CLK_ENABLE( );

        // Get the Oscillators configuration according to the internal RCC registers */
        HAL_RCC_GetOscConfig( &RCC_OscInitStruct );

        // Enable PLL
        RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_NONE;
        RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
        if( HAL_RCC_OscConfig( &RCC_OscInitStruct ) != HAL_OK )
        {
            while( 1 );
        }
        
        /* Get the Clocks configuration according to the internal RCC registers */
        HAL_RCC_GetClockConfig(&RCC_ClkInitStruct, &pFLatency);
        
        /* Select PLL as system clock source and keep HCLK, PCLK1 and PCLK2 clocks dividers as before */
        RCC_ClkInitStruct.ClockType     = RCC_CLOCKTYPE_SYSCLK;
        RCC_ClkInitStruct.SYSCLKSource  = RCC_SYSCLKSOURCE_PLLCLK;
        if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, pFLatency) != HAL_OK)
        {
            while(1);
        }
    }
    else
    {
        // MCU did not enter stop mode beacuse NVIC had a pending IT
    }

    CRITICAL_SECTION_END( );
}

/* SysTick_Handler is in Core/Src/stm32l4xx_it.c */
#if 0
void SysTick_Handler( void )
{
    HAL_IncTick( );
    HAL_SYSTICK_IRQHandler( );
}
#endif

uint8_t GetBoardPowerSource( void )
{
    if( UsbIsConnected == false )
    {
        return BATTERY_POWER;
    }
    else
    {
        return USB_POWER;
    }
}

/**
  * \brief Enters Low Power Stop Mode
  *
  * \note ARM exists the function when waking up
  */
void LpmEnterStopMode( void)
{
    CRITICAL_SECTION_BEGIN( );

    BoardDeInitMcu( );

    CRITICAL_SECTION_END( );

    // Enter Stop Mode
    HAL_PWR_EnterSTOPMode( PWR_LOWPOWERREGULATOR_ON, PWR_STOPENTRY_WFI );
}

/*!
 * \brief Exists Low Power Stop Mode
 */
void LpmExitStopMode( void )
{
    // Disable IRQ while the MCU is not running on HSI
    CRITICAL_SECTION_BEGIN( );

    // Initilizes the peripherals
    BoardInitMcu( );

    CRITICAL_SECTION_END( );
}

/*!
 * \brief Enters Low Power Sleep Mode
 *
 * \note ARM exits the function when waking up
 */
void LpmEnterSleepMode( void)
{
    HAL_PWR_EnterSLEEPMode(PWR_MAINREGULATOR_ON, PWR_SLEEPENTRY_WFI);
}

/*!
 * \brief Indicates if an erasing operation is on going.
 *
 * \remark This function is defined in eeprom-board.c file
 *
 * \retval isEradingOnGoing Returns true is an erasing operation is on going.
 */
bool EepromMcuIsErasingOnGoing( void );

void BoardLowPowerHandler( void )
{
    /* Custom board: sleep is managed entirely by main.c (Enter_Stop_Mode).
     * The Semtech stack must NOT enter low power on its own. */
    return;
}

#if 0 /* UART printf/scanf redirects not used on custom board */

/*
 * Function to be used by stdout for printf etc
 */
int _write( int fd, const void *buf, size_t count )
{
    while( UartPutBuffer( &Uart2, ( uint8_t* )buf, ( uint16_t )count ) != 0 ){ };
    return count;
}

/*
 * Function to be used by stdin for scanf etc
 */
int _read( int fd, const void *buf, size_t count )
{
    size_t bytesRead = 0;
    while( UartGetBuffer( &Uart2, ( uint8_t* )buf, count, ( uint16_t* )&bytesRead ) != 0 ){ };
    // Echo back the character
    while( UartPutBuffer( &Uart2, ( uint8_t* )buf, ( uint16_t )bytesRead ) != 0 ){ };
    return bytesRead;
}

/*
// Keil compiler
int fputc( int c, FILE *stream )
{
    while( UartPutChar( &Uart2, ( uint8_t )c ) != 0 );
    return c;
}

int fgetc( FILE *stream )
{
    uint8_t c = 0;
    while( UartGetChar( &Uart2, &c ) != 0 );
    // Echo back the character
    while( UartPutChar( &Uart2, c ) != 0 );
    return ( int )c;
}
*/

#endif

#ifdef USE_FULL_ASSERT

#include <stdio.h>

/*
 * Function Name  : assert_failed
 * Description    : Reports the name of the source file and the source line number
 *                  where the assert_param error has occurred.
 * Input          : - file: pointer to the source file name
 *                  - line: assert_param error line source number
 * Output         : None
 * Return         : None
 */
void assert_failed( uint8_t* file, uint32_t line )
{
    /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %lu\n", file, line) */

    printf( "Wrong parameters value: file %s on line %lu\n", ( const char* )file, line );
    /* Infinite loop */
    while( 1 )
    {
    }
}
#endif
