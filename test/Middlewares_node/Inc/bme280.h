/**
 ******************************************************************************
 * @file    bme280.h
 * @author  Daisy Berrios
 * @date    Jan 13, 2026
 * @brief   Driver for Bosch BME280 (Temp/Press/Humidity) - I2C (STM32 HAL)
 *******************************************************************************
 */
#ifndef _INC_BME280_H_
#define _INC_BME280_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <main.h>


/*- USER_Definitions --------------------------------------------------------*/
/* I2C address is 0x76 if SDO=GND, 0x77 if SDO=VDDIO (7-bit). Shifted for HAL. */
#ifndef BME280_I2C_ADDR
#define BME280_I2C_ADDR    (0x76 << 1)
#endif

#define BME280_CHIP_ID     0x60

/* Registers (see memory map) */
#define BME280_REG_ID         0xD0
#define BME280_REG_RESET      0xE0
#define BME280_REG_CTRL_HUM   0xF2
#define BME280_REG_STATUS     0xF3
#define BME280_REG_CTRL_MEAS  0xF4
#define BME280_REG_CONFIG     0xF5

#define BME280_REG_PRESS_MSB  0xF7
#define BME280_REG_TEMP_MSB   0xFA
#define BME280_REG_HUM_MSB    0xFD

/* Calibration blocks */
#define BME280_REG_CALIB00    0x88  /* 0x88..0xA1 */
#define BME280_REG_CALIB26    0xE1  /* 0xE1..0xE7 */

#define BME280_SOFTRESET_CMD  0xB6

/* Oversampling settings (osrs_x) */
typedef enum
{
  BME280_OSRS_SKIPPED = 0x00,
  BME280_OSRS_X1      = 0x01,
  BME280_OSRS_X2      = 0x02,
  BME280_OSRS_X4      = 0x03,
  BME280_OSRS_X8      = 0x04,
  BME280_OSRS_X16     = 0x05
} bme280_osrs_t;

/* Mode settings (mode[1:0]) */
typedef enum
{
  BME280_MODE_SLEEP  = 0x00,
  BME280_MODE_FORCED = 0x01, /* 01 or 10 are forced */
  BME280_MODE_NORMAL = 0x03
} bme280_mode_t;

/* IIR filter settings (filter[2:0]) */
typedef enum
{
  BME280_FILTER_OFF = 0x00,
  BME280_FILTER_2   = 0x01,
  BME280_FILTER_4   = 0x02,
  BME280_FILTER_8   = 0x03,
  BME280_FILTER_16  = 0x04
} bme280_filter_t;

/* Standby time settings (t_sb[2:0]) for normal mode */
typedef enum
{
  BME280_TSB_0P5MS   = 0x00,
  BME280_TSB_62P5MS  = 0x01,
  BME280_TSB_125MS   = 0x02,
  BME280_TSB_250MS   = 0x03,
  BME280_TSB_500MS   = 0x04,
  BME280_TSB_1000MS  = 0x05,
  BME280_TSB_10MS    = 0x06,
  BME280_TSB_20MS    = 0x07
} bme280_tsb_t;

/*- PUBLIC_Types ------------------------------------------------------------*/
typedef struct
{
  float temperature_c;
  float pressure_pa;
  float humidity_rh;
} bme280_data_t;

/*- PUBLIC_API --------------------------------------------------------------*/
void bme280_init(I2C_HandleTypeDef *hi2c);

/* Pre-canned low-power setup: forced mode, osrs 1x, filter off */
void bme280_config_weather_forced(I2C_HandleTypeDef *hi2c);

/* Triggers one forced measurement; call read after measuring bit clears */
void bme280_trigger_forced(I2C_HandleTypeDef *hi2c);

/* True while conversion running (status.measuring) */
bool bme280_is_measuring(I2C_HandleTypeDef *hi2c);

/* Reads + compensates T/P/H into float values */
bool bme280_read_compensated(I2C_HandleTypeDef *hi2c,bme280_data_t *out);

bool bme280_read_forced(I2C_HandleTypeDef *hi2c, bme280_data_t *out);

/* Optional placeholders to match your framework */
bool bme280_open(I2C_HandleTypeDef *hi2c);
bool bme280_update(I2C_HandleTypeDef *hi2c);

void bme280_trigger_forced(I2C_HandleTypeDef *hi2c);

#ifdef __cplusplus
}
#endif

#endif /* _INC_BME280_H_ */
