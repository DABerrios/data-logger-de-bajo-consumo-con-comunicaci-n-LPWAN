/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "fatfs.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "ff.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "LmHandler.h"
#include "LmhPackage.h"
#include "LmhpCompliance.h"
#include "timer.h"
#include "LoRaMac.h"
#include "LoRaMacCrypto.h"
#include "radio.h"
#include "sx126x.h"
#include "sx126x-board.h"
#include "rv3028.h"
#include "bme280.h"
#include "tmp117.h"
#include "diskio.h"
#include "functions.h"
#include "board.h"
#include "Commissioning.h"
#include "board-config.h"
#include "microshell.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
I2C_HandleTypeDef hi2c1;
I2C_HandleTypeDef hi2c2;

RTC_HandleTypeDef hrtc;

SPI_HandleTypeDef hspi1;
SPI_HandleTypeDef hspi2;

/* USER CODE BEGIN PV */
extern PCD_HandleTypeDef hpcd_USB_OTG_FS;
extern volatile bool rv3028_flag_update;
extern rv3028_t rtc_dev;

/* Interrupt flags set by EXTI callbacks, consumed by the main loop. */
volatile uint8_t g_rtc_irq = 0;
volatile uint8_t g_usb_irq = 0;
volatile uint8_t g_rain_irq = 0;
volatile uint32_t g_rain_pulses = 0;
volatile uint8_t g_sample_counter = 0;

/* LoRaWAN state */
static bool IsNetworkJoined = false;
static volatile uint8_t g_lora_powered = 0;
static volatile uint8_t g_lora_initialized = 0;

/* RTC wakeup interval in seconds (default 60 s, configurable via microshell).
 * LoRa TX occurs every g_lora_tx_every_n wakes.
 * At 60 s wake and N = 5, uplinks are sent every 5 minutes. */
static volatile uint32_t g_wake_interval_s = 60;
volatile uint32_t g_lora_tx_every_n = 1;
static volatile uint8_t  g_usb_connected   = 0;

/* LoRaWAN handler and parameter structures */
static LmHandlerCallbacks_t LmHandlerCallbacks = { .GetBatteryLevel = NULL,
		.GetTemperature = NULL, .GetRandomSeed = NULL, .OnMacProcess =
				OnMacProcess, .OnNvmDataChange = OnNvmDataChange,
		.OnNetworkParametersChange = OnNetworkParametersChange,
		.OnMacMcpsRequest = OnMacMcpsRequest, .OnMacMlmeRequest =
				OnMacMlmeRequest, .OnJoinRequest = OnJoinRequest, .OnTxData =
				OnTxData, .OnRxData = OnRxData, .OnClassChange = OnClassChange,
		.OnBeaconStatusChange = OnBeaconStatusChange, .OnSysTimeUpdate = NULL };

static LmHandlerParams_t LmHandlerParams = { .Region = ACTIVE_REGION,
		.AdrEnable = false, .IsTxConfirmed = LORAMAC_HANDLER_UNCONFIRMED_MSG,
		.TxDatarate = DR_6, .PublicNetworkEnable = LORAWAN_PUBLIC_NETWORK,
		.DutyCycleEnabled = false, .DataBufferMaxSize = 64, .DataBuffer = NULL,
		.PingSlotPeriodicity = 0 };
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_I2C2_Init(void);
static void MX_SPI1_Init(void);
static void MX_SPI2_Init(void);
static void MX_RTC_Init(void);
/* USER CODE BEGIN PFP */

void RadioOnDioIrq(void* context);
void SystemClock_Config_LowPower(void);
void Enter_Stop_Mode(void);
void Pins_Park(void);
static void rv3028_service_int(void);
static void I2C2_BusRecover(void);

/* LoRaWAN helpers (defined in functions.c) */
void ApplyAu915SubBand1Mask(void);

/* From sx126x-board.c */
void SX126xAntSwOff(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* ---------- Microshell hook implementations ----------------------------- */

/* Reads the current date and time from the external RV3028 RTC. */
static bool hook_rtc_get(microshell_datetime_t *out)
{
    rv3028_datetime_t dt;
    if (!rv3028_get_datetime(&rtc_dev, &dt)) return false;
    out->year  = 2000 + dt.year;
    out->month = dt.month;
    out->day   = dt.date;
    out->hour  = dt.hours;
    out->min   = dt.minutes;
    out->sec   = dt.seconds;
    return true;
}

static bool hook_rtc_set(const microshell_datetime_t *dt)
{
    rv3028_datetime_t rd = {0};
    rd.year    = (uint8_t)(dt->year >= 2000 ? dt->year - 2000 : dt->year);
    rd.month   = dt->month;
    rd.date    = dt->day;
    rd.hours   = dt->hour;
    rd.minutes = dt->min;
    rd.seconds = dt->sec;
    return rv3028_set_datetime(&rtc_dev, &rd);
}

/* Returns the current RTC wakeup interval in seconds. */
static bool hook_wakeup_get(uint32_t *out_s)
{
    *out_s = g_wake_interval_s;
    return true;
}

/* Reconfigures the RV3028 periodic wakeup timer.
 * Selects 1 Hz clock for intervals below 60 s, 1/60 Hz for longer periods. */
static bool hook_wakeup_set(uint32_t s)
{
    if (s == 0) return false;

    uint8_t  freq;
    uint16_t value;

    if (s < 60) {
        freq  = RV3028_CTRL1_TD_1HZ;
        value = (uint16_t)s;
    } else {
        uint32_t minutes = (s + 59) / 60;  /* round up */
        if (minutes > 4095) minutes = 4095;
        freq  = RV3028_CTRL1_TD_60S;
        value = (uint16_t)minutes;
        s = minutes * 60;  /* store actual interval */
    }

    if (!rv3028_timer_set(&rtc_dev, value, freq, true)) return false;
    g_wake_interval_s = s;

    microshell_printf("Wake interval set to %lu s\r\n", (unsigned long)s);
    return true;
}

/* Returns the TX interval in seconds (wake count multiplied by wake period). */
static bool hook_txint_get(uint32_t *out_s)
{
    *out_s = g_lora_tx_every_n * g_wake_interval_s;
    return true;
}

static bool hook_txint_set(uint32_t s)
{
    if (s == 0 || g_wake_interval_s == 0) return false;

    uint32_t n = s / g_wake_interval_s;
    if (n == 0) n = 1;
    g_lora_tx_every_n = n;

    microshell_printf("TX every %lu wakes (%lu s)\r\n",
                      (unsigned long)n, (unsigned long)(n * g_wake_interval_s));
    return true;
}

/* Handles custom microshell commands: status, txtest, and regs. */
static bool hook_custom(int argc, char **argv)
{
    if (argc >= 1 && strcmp(argv[0], "status") == 0)
    {
        microshell_printf("sample_cnt=%u tx_every=%lu wake_s=%lu\r\n",
                          (unsigned)g_sample_counter,
                          (unsigned long)g_lora_tx_every_n,
                          (unsigned long)g_wake_interval_s);
        microshell_printf("lora_init=%u lora_pwr=%u usb_det=%u\r\n",
                          (unsigned)g_lora_initialized,
                          (unsigned)g_lora_powered,
                          (unsigned)g_usb_connected);
        return true;
    }

    if (argc >= 1 && strcmp(argv[0], "txtest") == 0)
    {
        if (!g_lora_initialized) {
            microshell_print("ERR: LoRa not initialized\r\n");
            return true;
        }

        microshell_print("TX test: init radio...\r\n");

        /* Power the radio module and perform a full cold-start re-init. */
        V_TRANS_On();
        HAL_Delay(20);
        g_lora_powered = 1;
        BoardRadioPowerCycleInit();
        ApplyAu915SubBand1Mask();

        /* Build and transmit a fixed diagnostic payload. */
        uint8_t testBuf[7] = { 'T', 'E', 'S', 'T', ':', 0x01, 0x02 };
        LmHandlerAppData_t testData = { 0 };
        testData.Buffer  = testBuf;
        testData.BufferSize = 7;
        testData.Port    = 2;

        LmHandlerSend(&testData, LORAMAC_HANDLER_UNCONFIRMED_MSG);
        microshell_print("TX queued, processing...\r\n");

        /* Process the MAC for the full TX + RX window cycle. */
        uint32_t t0 = HAL_GetTick();
        while ((HAL_GetTick() - t0) < 10000) {
            LmHandlerProcess();
            HAL_Delay(1);
        }

        SX126xAntSwOff();
        HAL_SPI_DeInit(&hspi1);
        g_lora_powered = 0;
        microshell_print("TX test done\r\n");
        return true;
    }

    if (argc >= 1 && strcmp(argv[0], "regs") == 0)
    {
        if (!g_lora_initialized) {
            microshell_print("ERR: LoRa not initialized\r\n");
            return true;
        }

        /* Power the radio and read DIO3 and crystal trim registers via SPI. */
        V_TRANS_On();
        HAL_Delay(20);
        g_lora_powered = 1;
        BoardRadioPowerCycleInit();

        extern uint8_t SX126xReadRegister( uint16_t address );
        uint8_t r0580 = SX126xReadRegister(0x0580);
        uint8_t r0583 = SX126xReadRegister(0x0583);
        uint8_t r0584 = SX126xReadRegister(0x0584);
        uint8_t r0585 = SX126xReadRegister(0x0585);
        uint8_t r0920 = SX126xReadRegister(0x0920);
        uint8_t r0911 = SX126xReadRegister(0x0911);
        uint8_t r0912 = SX126xReadRegister(0x0912);
        uint8_t rxg   = SX126xReadRegister(0x08AC);

        microshell_printf("DIO3: 0580=%02X 0583=%02X 0584=%02X 0585=%02X\r\n",
                          r0580, r0583, r0584, r0585);
        microshell_printf("0920=%02X  XTAL: 0911=%02X 0912=%02X  RXG=%02X\r\n",
                          r0920, r0911, r0912, rxg);

        HAL_SPI_DeInit(&hspi1);
        g_lora_powered = 0;
        return true;
    }

    return false;  /* not handled */
}

static const microshell_hooks_t ms_hooks = {
    .rtc_get     = hook_rtc_get,
    .rtc_set     = hook_rtc_set,
    .wakeup_get_s = hook_wakeup_get,
    .wakeup_set_s = hook_wakeup_set,
    .txint_get_s  = hook_txint_get,
    .txint_set_s  = hook_txint_set,
    .data_list    = NULL,
    .data_read    = NULL,
    .data_clear   = NULL,
    .custom_line  = hook_custom,
};

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C1_Init();
  MX_I2C2_Init();
  MX_SPI1_Init();
  MX_SPI2_Init();
  MX_RTC_Init();
  MX_FATFS_Init();
  MX_USB_DEVICE_Init();
  /* USER CODE BEGIN 2 */

	/* 1. Power sensor and transceiver rails, initialise the external RTC. */
	V_SENSOR_On();
	V_TRANS_On();
	HAL_Delay(20);
	rv3028_init(&rtc_dev, &hi2c2, RV3028_I2C_ADDR_DEFAULT);

	/* Set the initial date and time on the external RTC. */
	rv3028_datetime_t startTime = { .year = 2026, .month = 4, .date = 8,
			.hours = 10, .minutes = 35, .seconds = 0, .weekday = 5 };
	rv3028_set_datetime(&rtc_dev, &startTime);

	/* 2. Configure the RV3028 periodic countdown timer for wakeup.
	 *    The timer fires every 60 s (1-minute resolution). */
	rv3028_timer_set(&rtc_dev, 1, RV3028_CTRL1_TD_60S, true);
	rv3028_service_int();
	__HAL_GPIO_EXTI_CLEAR_IT(RTC_int_Pin);

	rv3028_datetime_t now;
	bme280_data_t bme;
	float t_c = 0.0f;
	rv3028_get_datetime(&rtc_dev, &now);

	tmp117_init(&hi2c2);
	bme280_init(&hi2c2);
	HAL_Delay(20);
	bme280_read_forced(&hi2c2, &bme);
	tmp117_oneshot_read_temperature_c(&hi2c2, &t_c, 200);

	/* 3. Allow time for USB enumeration and microshell access. */
	HAL_Delay(5000);

	/* 4. LoRaWAN ABP initialisation (one-time, before first sleep).
	 *    V_SENSOR supplies the LoRa module (V_TRANS TPS22916 is damaged).
	 *    BoardInitMcu configures SPI1 and the radio GPIO pins. */
	__HAL_RCC_CLEAR_RESET_FLAGS();

	BoardInitMcu();

	if (LmHandlerInit(&LmHandlerCallbacks, &LmHandlerParams)
			== LORAMAC_HANDLER_SUCCESS) {

		LmHandlerSetSystemMaxRxError(50);
		ApplyAu915SubBand1Mask();
		LmHandlerJoin();
		ApplyAu915SubBand1Mask();

		/* Force DR_6 (SF8 / 500 kHz), disable ADR, limit to one transmission. */
		{
			MibRequestConfirm_t mibReq;
			mibReq.Type = MIB_CHANNELS_DATARATE;
			mibReq.Param.ChannelsDatarate = DR_6;
			LoRaMacMibSetRequestConfirm(&mibReq);

			mibReq.Type = MIB_CHANNELS_DEFAULT_DATARATE;
			mibReq.Param.ChannelsDefaultDatarate = DR_6;
			LoRaMacMibSetRequestConfirm(&mibReq);

			mibReq.Type = MIB_ADR;
			mibReq.Param.AdrEnable = false;
			LoRaMacMibSetRequestConfirm(&mibReq);

			mibReq.Type = MIB_CHANNELS_NB_TRANS;
			mibReq.Param.ChannelsNbTrans = 1;
			LoRaMacMibSetRequestConfirm(&mibReq);
		}

		/* Restore the uplink frame counter from the RTC backup register.
		 * This value persists across STOP2 sleep and soft resets. */
		{
			uint32_t savedFCnt = HAL_RTCEx_BKUPRead(&hrtc, RTC_BKP_DR4);
			if (savedFCnt > 0) {
				LoRaMacCryptoSetFCntUp(savedFCnt);
			}
		}

		g_lora_initialized = 1;
		IsNetworkJoined = true;
	}

	/* 5. Transition to low-power mode.
	 *    Clear RTC flags and de-initialise all peripherals to prevent
	 *    current leakage through inactive bus lines. */
	rv3028_clear_flags(&rtc_dev, RV3028_FLAG_TF);
	g_rtc_irq = 0;
	g_sample_counter = 0;

	HAL_SPI_DeInit(&hspi1);
	HAL_SPI_DeInit(&hspi2);
	HAL_I2C_DeInit(&hi2c2);
	Pins_Park();
	V_SENSOR_Off();
	V_TRANS_Off();
	g_lora_powered = 0;
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	while (1) {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
		if (g_rtc_irq) {
			g_rtc_irq = 0;
			sd_invalidate();
			g_sample_counter++;

			/* Restore peripheral power and re-enable the SD chip-select GPIO. */
			V_SENSOR_On();
			HAL_Delay(100);
			__HAL_RCC_GPIOB_CLK_ENABLE();
			GPIO_InitTypeDef cs_init = {0};
			cs_init.Pin = SD_CS_Pin;  /* PB12 */
			cs_init.Mode = GPIO_MODE_OUTPUT_PP;
			cs_init.Pull = GPIO_NOPULL;
			cs_init.Speed = GPIO_SPEED_FREQ_LOW;
			HAL_GPIO_Init(SD_CS_GPIO_Port, &cs_init);

			HAL_GPIO_WritePin(SD_CS_GPIO_Port, SD_CS_Pin, GPIO_PIN_SET);

			HAL_I2C_DeInit(&hi2c2);
			HAL_SPI_DeInit(&hspi2);
			I2C2_BusRecover();

			/* Reset I2C2 and SPI2 hardware blocks after power-cycle wake. */
			__HAL_RCC_I2C2_FORCE_RESET();
			HAL_Delay(1);
			__HAL_RCC_I2C2_RELEASE_RESET();

			__HAL_RCC_SPI2_FORCE_RESET();
			HAL_Delay(1);
			__HAL_RCC_SPI2_RELEASE_RESET();

			/* Re-initialise peripherals for sensor and SD card access. */
			MX_I2C2_Init();
			MX_SPI2_Init();
			MX_RTC_Init();

			/* Read all sensors and append the sample to DATA.csv on the SD card.
			 * send_lora_payload() later reads the last N lines for averaging. */
			if (rv3028_wakeup_init(&rtc_dev)) {
				bme280_init(&hi2c2);
				tmp117_init(&hi2c2);
				log_one_sample();
			}

			/* LoRaWAN uplink (every N wake cycles). */
			if (g_lora_initialized
					&& (g_sample_counter >= g_lora_tx_every_n)) {
				g_sample_counter = 0;

				/* Power the LoRa module and perform a full radio cold-start. */
				V_TRANS_On();
				HAL_Delay(20);
				g_lora_powered = 1;
				BoardRadioPowerCycleInit();

				/* Re-apply channel mask and single-TX settings. */
				ApplyAu915SubBand1Mask();
				{
					MibRequestConfirm_t m;
					m.Type = MIB_CHANNELS_NB_TRANS;
					m.Param.ChannelsNbTrans = 1;
					LoRaMacMibSetRequestConfirm(&m);

					m.Type = MIB_CHANNELS_DATARATE;
					m.Param.ChannelsDatarate = DR_6;
					LoRaMacMibSetRequestConfirm(&m);
				}

				/* Transmit the averaged sensor data and rain accumulator. */
				send_lora_payload();

				/* TX-only fire-and-forget: poll 200 ms for RadioIrqProcess to
				 * consume TxDone, then kill RX window timers and force the
				 * MAC state machine back to IDLE. */
				{
					uint32_t t0 = HAL_GetTick();
					while ((HAL_GetTick() - t0) < 200) {
						LmHandlerProcess();
						HAL_Delay(1);
					}
				}
				TimerStopAll();
				LoRaMacStart();

				/* Persist the uplink frame counter in the RTC backup register. */
				{
					uint32_t fCnt;
					LoRaMacCryptoGetFCntUp(&fCnt);
					HAL_RTCEx_BKUPWrite(&hrtc, RTC_BKP_DR4, fCnt);
				}

				/* Power down the LoRa module. */
				SX126xAntSwOff();
				HAL_SPI_DeInit(&hspi1);
				g_lora_powered = 0;
			}

			/* Clear the RTC timer flag to release the interrupt line. */
			rv3028_clear_flags(&rtc_dev, RV3028_FLAG_TF);

			/* Shut down all peripherals and power rails before sleep. */
			g_rtc_irq = 0;
			HAL_SPI_DeInit(&hspi1);
			HAL_SPI_DeInit(&hspi2);
			HAL_I2C_DeInit(&hi2c2);
			HAL_I2C_DeInit(&hi2c1);
			__HAL_RCC_SPI1_CLK_DISABLE();
			__HAL_RCC_SPI2_CLK_DISABLE();
			__HAL_RCC_I2C2_CLK_DISABLE();
			__HAL_RCC_I2C1_CLK_DISABLE();
			Pins_Park();
			V_SENSOR_Off();
			V_TRANS_Off();
			g_lora_powered = 0;
			HAL_RTC_DeInit(&hrtc);

			Enter_Stop_Mode();

		} else {
			if (hrtc.State != HAL_RTC_STATE_RESET) {
				HAL_RTC_DeInit(&hrtc);
			}
			Enter_Stop_Mode();
		}
		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */

	}
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure LSE Drive Capability
  */
  HAL_PWR_EnableBkUpAccess();
  __HAL_RCC_LSEDRIVE_CONFIG(RCC_LSEDRIVE_LOW);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_LSE|RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.LSEState = RCC_LSE_ON;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = 0;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 24;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Enable MSI Auto calibration
  */
  HAL_RCCEx_EnableMSIPLLMode();
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = 0x10805D88;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief I2C2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C2_Init(void)
{

  /* USER CODE BEGIN I2C2_Init 0 */

  /* USER CODE END I2C2_Init 0 */

  /* USER CODE BEGIN I2C2_Init 1 */

  /* USER CODE END I2C2_Init 1 */
  hi2c2.Instance = I2C2;
  hi2c2.Init.Timing = 0x10805D88;
  hi2c2.Init.OwnAddress1 = 0;
  hi2c2.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c2.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c2.Init.OwnAddress2 = 0;
  hi2c2.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c2.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c2.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c2, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c2, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C2_Init 2 */

  /* USER CODE END I2C2_Init 2 */

}

/**
  * @brief RTC Initialization Function
  * @param None
  * @retval None
  */
static void MX_RTC_Init(void)
{

  /* USER CODE BEGIN RTC_Init 0 */

  /* USER CODE END RTC_Init 0 */

  /* USER CODE BEGIN RTC_Init 1 */

  /* USER CODE END RTC_Init 1 */

  /** Initialize RTC Only
  */
  hrtc.Instance = RTC;
  hrtc.Init.HourFormat = RTC_HOURFORMAT_24;
  hrtc.Init.AsynchPrediv = 127;
  hrtc.Init.SynchPrediv = 255;
  hrtc.Init.OutPut = RTC_OUTPUT_DISABLE;
  hrtc.Init.OutPutRemap = RTC_OUTPUT_REMAP_NONE;
  hrtc.Init.OutPutPolarity = RTC_OUTPUT_POLARITY_HIGH;
  hrtc.Init.OutPutType = RTC_OUTPUT_TYPE_OPENDRAIN;
  if (HAL_RTC_Init(&hrtc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN RTC_Init 2 */

  /* USER CODE END RTC_Init 2 */

}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 7;
  hspi1.Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
  hspi1.Init.NSSPMode = SPI_NSS_PULSE_DISABLE;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief SPI2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI2_Init(void)
{

  /* USER CODE BEGIN SPI2_Init 0 */

  /* USER CODE END SPI2_Init 0 */

  /* USER CODE BEGIN SPI2_Init 1 */

  /* USER CODE END SPI2_Init 1 */
  /* SPI2 parameter configuration*/
  hspi2.Instance = SPI2;
  hspi2.Init.Mode = SPI_MODE_MASTER;
  hspi2.Init.Direction = SPI_DIRECTION_2LINES;
  hspi2.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi2.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi2.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi2.Init.NSS = SPI_NSS_SOFT;
  hspi2.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
  hspi2.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi2.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi2.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi2.Init.CRCPolynomial = 7;
  hspi2.Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
  hspi2.Init.NSSPMode = SPI_NSS_PULSE_DISABLE;
  if (HAL_SPI_Init(&hspi2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI2_Init 2 */

  /* USER CODE END SPI2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, EN_v_trans_Pin|EN_v_sensor_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, LORA_CS_Pin|LORA_rst_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(Lora_ANT_SW_GPIO_Port, Lora_ANT_SW_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(SD_CS_GPIO_Port, SD_CS_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(BQ25570_Vout_EN_GPIO_Port, BQ25570_Vout_EN_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : rain_int_Pin */
  GPIO_InitStruct.Pin = rain_int_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(rain_int_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PC0 PC1 PC3 PC6
                           PC7 PC8 PC9 PC11
                           PC12 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_3|GPIO_PIN_6
                          |GPIO_PIN_7|GPIO_PIN_8|GPIO_PIN_9|GPIO_PIN_11
                          |GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : PA0 PA3 PA4 PA10 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_3|GPIO_PIN_4|GPIO_PIN_10;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : EN_v_trans_Pin EN_v_sensor_Pin */
  GPIO_InitStruct.Pin = EN_v_trans_Pin|EN_v_sensor_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : LORA_CS_Pin LORA_rst_Pin BQ25570_Vout_EN_Pin */
  GPIO_InitStruct.Pin = LORA_CS_Pin|LORA_rst_Pin|BQ25570_Vout_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : LORA_DIO_Pin usb_wakeup_Pin */
  GPIO_InitStruct.Pin = LORA_DIO_Pin|usb_wakeup_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : LORA_BUSY_Pin */
  GPIO_InitStruct.Pin = LORA_BUSY_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(LORA_BUSY_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : Lora_ANT_SW_Pin SD_CS_Pin */
  GPIO_InitStruct.Pin = Lora_ANT_SW_Pin|SD_CS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : PB12 PB5 PB7 (analog) */
  GPIO_InitStruct.Pin = GPIO_PIN_12|GPIO_PIN_5|GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : usb_detect_Pin (PB4) - USB cable detect interrupt */
  GPIO_InitStruct.Pin = usb_detect_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(usb_detect_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : RTC_int_Pin */
  GPIO_InitStruct.Pin = RTC_int_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(RTC_int_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : BQ25570_VBATok_Pin */
  GPIO_InitStruct.Pin = BQ25570_VBATok_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(BQ25570_VBATok_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PD2 */
  GPIO_InitStruct.Pin = GPIO_PIN_2;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI0_IRQn);

  HAL_NVIC_SetPriority(EXTI4_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(EXTI4_IRQn);

  HAL_NVIC_SetPriority(EXTI9_5_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI9_5_IRQn);

  HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* ---------- EXTI callback and IRQ handlers ------------------------------ */

/** Dispatches GPIO EXTI events to the appropriate interrupt flags. */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
	if (GPIO_Pin == usb_wakeup_Pin) {
		g_usb_irq = 1;
	}
	if (GPIO_Pin == usb_detect_Pin) {
		g_usb_connected = 1;
	}
	if (GPIO_Pin == RTC_int_Pin) {
		g_rtc_irq = 1;
	}
	if (GPIO_Pin == rain_int_Pin) {
		g_rain_pulses++;
	}
}

/** EXTI0 handles PB0 (LoRa DIO1). Forwards the IRQ to the radio driver
 *  when the module is powered. */
void EXTI0_IRQHandler(void) {
	HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_0);
    if (g_lora_powered) {
        RadioOnDioIrq( NULL );
    }
}

/** EXTI4 handles PB4 (USB cable detect). */
void EXTI4_IRQHandler(void) {
	HAL_GPIO_EXTI_IRQHandler(usb_detect_Pin);
}

/** EXTI9..5 handles PB5 (USB wakeup) and PA8 (RTC interrupt). */
void EXTI9_5_IRQHandler(void) {
	HAL_GPIO_EXTI_IRQHandler(usb_wakeup_Pin);
	HAL_GPIO_EXTI_IRQHandler(RTC_int_Pin);
}

/** EXTI15..10 handles PC13 (rain gauge tipping bucket). */
void EXTI15_10_IRQHandler(void) {
	HAL_GPIO_EXTI_IRQHandler(rain_int_Pin);
}

/* ---------- Clock and power management ---------------------------------- */

/** Switches the system clock to MSI 4 MHz with PLL disabled.
 *  Called before entering STOP2 and immediately after wakeup. */
void SystemClock_Config_LowPower(void) {
	RCC_OscInitTypeDef RCC_OscInitStruct = { 0 };
	RCC_ClkInitTypeDef RCC_ClkInitStruct = { 0 };

	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI;
	RCC_OscInitStruct.MSIState = RCC_MSI_ON;
	RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;  /* 4 MHz */
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_OFF;
	HAL_RCC_OscConfig(&RCC_OscInitStruct);

	RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
			| RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_MSI;
	HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0);

	SystemCoreClockUpdate();
}
/** Forces PA0 (PMIC enable) low to prevent back-powering during sleep. */
static void PMIC_Pin_ForceLow(void) {
	__HAL_RCC_GPIOA_CLK_ENABLE();
	GPIO_InitTypeDef g = {0};
	g.Pin   = GPIO_PIN_0;           /* PA0 */
	g.Mode  = GPIO_MODE_OUTPUT_PP;
	g.Pull  = GPIO_PULLDOWN;
	g.Speed = GPIO_SPEED_FREQ_LOW;
	HAL_GPIO_Init(GPIOA, &g);
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0, GPIO_PIN_RESET);
}

/** Enters STM32L4 STOP2 mode. Clears all pending EXTI flags before sleep
 *  and reconfigures the clock to MSI 4 MHz on wake. */
void Enter_Stop_Mode(void) {
	PMIC_Pin_ForceLow();
	SystemClock_Config_LowPower();
	HAL_SuspendTick();
	__HAL_PWR_CLEAR_FLAG(PWR_FLAG_WU);
	__HAL_GPIO_EXTI_CLEAR_IT(RTC_int_Pin);
	__HAL_GPIO_EXTI_CLEAR_IT(GPIO_PIN_0);
	__HAL_GPIO_EXTI_CLEAR_IT(GPIO_PIN_13);
	__HAL_PWR_CLEAR_FLAG(PWR_FLAG_WU);

	HAL_PWREx_EnterSTOP2Mode(PWR_STOPENTRY_WFI);
	SystemClock_Config_LowPower();
	HAL_ResumeTick();
}


/** Parks all GPIO pins for minimum leakage before STOP2 sleep.
 *
 *  Protected pins (not touched):
 *    PA8  - RTC external interrupt (must remain active during sleep)
 *    PA13 - SWDIO  (debug access)
 *    PA14 - SWCLK  (debug access)
 *    PC13 - Rain gauge interrupt (must remain active during sleep)
 *    PC14 - LSE crystal
 *    PC15 - LSE crystal
 *
 *  All other pins are set to analog (high-Z) to prevent back-feed.
 *  Power-control pins are explicitly driven low to disable supply rails.
 */
void Pins_Park(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    GPIO_InitStruct.Pull = GPIO_NOPULL;

    /* Port A: all except PA8 (RTC INT), PA13/PA14 (SWD). */
    GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2 |
                          GPIO_PIN_3 | GPIO_PIN_4 |
                          GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7 |
                          GPIO_PIN_9 | GPIO_PIN_10 |
                          GPIO_PIN_11 | GPIO_PIN_12 |
                          GPIO_PIN_15;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* Port B: all 16 pins. */
    GPIO_InitStruct.Pin = GPIO_PIN_All;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /* Port C: all except PC13 (rain INT), PC14/PC15 (LSE). */
    GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3 |
                          GPIO_PIN_4 | GPIO_PIN_5 |
                          GPIO_PIN_6 | GPIO_PIN_7 |
                          GPIO_PIN_8 | GPIO_PIN_9 |
                          GPIO_PIN_10 | GPIO_PIN_11 | GPIO_PIN_12;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    /* Port D. */
    GPIO_InitStruct.Pin = GPIO_PIN_2;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

    /* Drive power-control pins low to disable supply rails. */
    GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull  = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

    /* V_TRANS (PA1) and V_SENSOR (PA2). */
    GPIO_InitStruct.Pin = GPIO_PIN_1 | GPIO_PIN_2;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_1 | GPIO_PIN_2, GPIO_PIN_RESET);

    /* BQ25570 output enable (PC10). */
    GPIO_InitStruct.Pin = GPIO_PIN_10;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_10, GPIO_PIN_RESET);
}

/** Reads and clears all pending RV3028 status flags to release the INT line. */
static void rv3028_service_int(void) {
	uint8_t st = 0;

	if (!rv3028_get_flags(&rtc_dev, &st)) {
		return;
	}

	uint8_t clr = 0;
	if (st & RV3028_FLAG_TF)
		clr |= RV3028_FLAG_TF;
	if (st & RV3028_FLAG_AF)
		clr |= RV3028_FLAG_AF;
	if (st & RV3028_FLAG_EVF)
		clr |= RV3028_FLAG_EVF;
	if (st & RV3028_FLAG_UF)
		clr |= RV3028_FLAG_UF;
	if (st & RV3028_FLAG_BSF)
		clr |= RV3028_FLAG_BSF;
	if (st & RV3028_FLAG_CLKF)
		clr |= RV3028_FLAG_CLKF;
	if (st & RV3028_FLAG_PORF)
		clr |= RV3028_FLAG_PORF;

	if (clr)
		rv3028_clear_flags(&rtc_dev, clr);
}

/** Recovers the I2C2 bus if a slave device holds SDA low after power cycle.
 *  Generates up to nine clock pulses followed by a START/STOP sequence to
 *  reset the slave state machine. Open-drain mode is required. */
static void I2C2_BusRecover(void) {
	GPIO_InitTypeDef GPIO_InitStruct = { 0 };
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	GPIO_InitStruct.Pin = GPIO_PIN_13 | GPIO_PIN_14;  /* SCL | SDA */
	HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

	/* Release both lines to idle state. */
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_13, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_SET);
    HAL_Delay(1);

	/* Clock up to nine pulses until the slave releases SDA. */
	for (int i = 0; i < 9; i++) {
		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_13, GPIO_PIN_RESET);
		HAL_Delay(1);
		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_13, GPIO_PIN_SET);
		HAL_Delay(1);

		if (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_14) == GPIO_PIN_SET) {
			break;
		}
	}

	/* Generate a START then STOP condition to reset the slave. */
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);
    HAL_Delay(1);
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_13, GPIO_PIN_SET);
    HAL_Delay(1);
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_SET);
    HAL_Delay(1);
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1) {
	}
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
