/**
 ******************************************************************************
 * @file    bme280.c
 * @author  Daisy Berrios
 * @date    Jan 13, 2026
 * @brief   Driver for Bosch BME280 (Temp/Press/Humidity) - I2C (STM32 HAL)
 *
 * @note    Uses burst read for data registers (0xF7..0xFE) for consistency.
 * @note    Reads trimming parameters from 0x88..0xA1 and 0xE1..0xE7.
 *******************************************************************************
 */

#include <string.h>
#include "bme280.h"
#include "main.h"


/*- PRIVATE_Definitions ------------------------------------------------------*/
#define TIMEOUT_100MS 100

/*- PRIVATE_Types ------------------------------------------------------------*/
typedef struct
{
  /* Temperature calibration */
  uint16_t dig_T1;
  int16_t  dig_T2;
  int16_t  dig_T3;

  /* Pressure calibration */
  uint16_t dig_P1;
  int16_t  dig_P2;
  int16_t  dig_P3;
  int16_t  dig_P4;
  int16_t  dig_P5;
  int16_t  dig_P6;
  int16_t  dig_P7;
  int16_t  dig_P8;
  int16_t  dig_P9;

  /* Humidity calibration */
  uint8_t  dig_H1;
  int16_t  dig_H2;
  uint8_t  dig_H3;
  int16_t  dig_H4;
  int16_t  dig_H5;
  int8_t   dig_H6;

  int32_t  t_fine;
  bool     calib_loaded;
} bme280_calib_t;

/*- PRIVATE_Data -------------------------------------------------------------*/
static bme280_calib_t s_cal = {0};

/*- PRIVATE_Functions --------------------------------------------------------*/
static void bme280_i2c_read(I2C_HandleTypeDef *hi2c,uint8_t reg, uint8_t *buf, uint16_t len)
{
  if (HAL_I2C_Mem_Read(hi2c, BME280_I2C_ADDR, reg, 1, buf, len, TIMEOUT_100MS) != HAL_OK)
  {
    Error_Handler();
  }
}

static void bme280_i2c_write(I2C_HandleTypeDef *hi2c,uint8_t reg, const uint8_t *buf, uint16_t len)
{
  if (HAL_I2C_Mem_Write(hi2c, BME280_I2C_ADDR, reg, 1, (uint8_t*)buf, len, TIMEOUT_100MS) != HAL_OK)
  {
    Error_Handler();
  }
}

static uint8_t bme280_read_u8(I2C_HandleTypeDef *hi2c,uint8_t reg)
{
  uint8_t v = 0;
  bme280_i2c_read(hi2c,reg, &v, 1);
  return v;
}

static void bme280_write_u8(I2C_HandleTypeDef *hi2c,uint8_t reg, uint8_t v)
{
  bme280_i2c_write(hi2c,reg, &v, 1);
}

static int32_t bme280_compensate_T_int32(I2C_HandleTypeDef *hi2c,int32_t adc_T)
{
  int32_t var1, var2, T;

  var1 = ((((adc_T >> 3) - ((int32_t)s_cal.dig_T1 << 1))) * ((int32_t)s_cal.dig_T2)) >> 11;
  var2 = (((((adc_T >> 4) - ((int32_t)s_cal.dig_T1)) * ((adc_T >> 4) - ((int32_t)s_cal.dig_T1))) >> 12) *
           ((int32_t)s_cal.dig_T3)) >> 14;

  s_cal.t_fine = var1 + var2;
  T = (s_cal.t_fine * 5 + 128) >> 8; // 0.01 °C

  return T;
}

static uint32_t bme280_compensate_P_int64(I2C_HandleTypeDef *hi2c,int32_t adc_P)
{
  int64_t var1, var2, p;

  var1 = ((int64_t)s_cal.t_fine) - 128000;
  var2 = var1 * var1 * (int64_t)s_cal.dig_P6;
  var2 = var2 + ((var1 * (int64_t)s_cal.dig_P5) << 17);
  var2 = var2 + (((int64_t)s_cal.dig_P4) << 35);
  var1 = ((var1 * var1 * (int64_t)s_cal.dig_P3) >> 8) + ((var1 * (int64_t)s_cal.dig_P2) << 12);
  var1 = (((((int64_t)1) << 47) + var1) * (int64_t)s_cal.dig_P1) >> 33;

  if (var1 == 0) return 0; // avoid div0

  p = 1048576 - adc_P;
  p = (((p << 31) - var2) * 3125) / var1;
  var1 = ((int64_t)s_cal.dig_P9 * (p >> 13) * (p >> 13)) >> 25;
  var2 = ((int64_t)s_cal.dig_P8 * p) >> 19;
  p = ((p + var1 + var2) >> 8) + (((int64_t)s_cal.dig_P7) << 4);

  return (uint32_t)p; // Q24.8 Pa
}

static uint32_t bme280_compensate_H_int32(I2C_HandleTypeDef *hi2c,int32_t adc_H)
{
  int32_t v_x1_u32r;

  v_x1_u32r = (s_cal.t_fine - ((int32_t)76800));
  v_x1_u32r = (((((adc_H << 14) - (((int32_t)s_cal.dig_H4) << 20) - (((int32_t)s_cal.dig_H5) * v_x1_u32r)) +
                ((int32_t)16384)) >> 15) *
              (((((((v_x1_u32r * ((int32_t)s_cal.dig_H6)) >> 10) *
                   (((v_x1_u32r * ((int32_t)s_cal.dig_H3)) >> 11) + ((int32_t)32768))) >> 10) +
                 ((int32_t)2097152)) *
                ((int32_t)s_cal.dig_H2) + 8192) >> 14));

  v_x1_u32r = (v_x1_u32r - (((((v_x1_u32r >> 15) * (v_x1_u32r >> 15)) >> 7) * ((int32_t)s_cal.dig_H1)) >> 4));
  v_x1_u32r = (v_x1_u32r < 0 ? 0 : v_x1_u32r);
  v_x1_u32r = (v_x1_u32r > 419430400 ? 419430400 : v_x1_u32r);

  return (uint32_t)(v_x1_u32r >> 12); // Q22.10 %RH
}

static void bme280_read_calibration(I2C_HandleTypeDef *hi2c)
{
  uint8_t buf1[26] = {0};
  uint8_t buf2[7]  = {0};

  /* calib00..calib25 at 0x88..0xA1 */
  bme280_i2c_read(hi2c,BME280_REG_CALIB00, buf1, sizeof(buf1));
  /* calib26..calib41 at 0xE1..0xE7 (we only need 0xE1..0xE7) */
  bme280_i2c_read(hi2c,BME280_REG_CALIB26, buf2, sizeof(buf2));

  s_cal.dig_T1 = (uint16_t)(buf1[1]  << 8) | buf1[0];
  s_cal.dig_T2 = (int16_t) ((buf1[3]  << 8) | buf1[2]);
  s_cal.dig_T3 = (int16_t) ((buf1[5]  << 8) | buf1[4]);

  s_cal.dig_P1 = (uint16_t)(buf1[7]  << 8) | buf1[6];
  s_cal.dig_P2 = (int16_t) ((buf1[9]  << 8) | buf1[8]);
  s_cal.dig_P3 = (int16_t) ((buf1[11] << 8) | buf1[10]);
  s_cal.dig_P4 = (int16_t) ((buf1[13] << 8) | buf1[12]);
  s_cal.dig_P5 = (int16_t) ((buf1[15] << 8) | buf1[14]);
  s_cal.dig_P6 = (int16_t) ((buf1[17] << 8) | buf1[16]);
  s_cal.dig_P7 = (int16_t) ((buf1[19] << 8) | buf1[18]);
  s_cal.dig_P8 = (int16_t) ((buf1[21] << 8) | buf1[20]);
  s_cal.dig_P9 = (int16_t) ((buf1[23] << 8) | buf1[22]);

  s_cal.dig_H1 = buf1[25];

  s_cal.dig_H2 = (int16_t)((buf2[1] << 8) | buf2[0]);
  s_cal.dig_H3 = buf2[2];

  /* H4/H5 are split across bytes (see datasheet mapping) */
  s_cal.dig_H4 = (int16_t)((buf2[3] << 4) | (buf2[4] & 0x0F));
  s_cal.dig_H5 = (int16_t)((buf2[5] << 4) | (buf2[4] >> 4));
  s_cal.dig_H6 = (int8_t)buf2[6];

  s_cal.calib_loaded = true;
}

/*- PUBLIC_API ---------------------------------------------------------------*/
void bme280_init(I2C_HandleTypeDef *hi2c)
{
  uint8_t id = bme280_read_u8(hi2c,BME280_REG_ID);
  if (id != BME280_CHIP_ID)
  {
    Error_Handler();
    return;
  }

  /* Soft reset: write 0xB6 to 0xE0 */
  bme280_write_u8(hi2c,BME280_REG_RESET, BME280_SOFTRESET_CMD);

  /* Give sensor a little time to copy NVM (im_update bit clears) */
  while (bme280_read_u8(hi2c, BME280_REG_STATUS) & 0x01) {
    HAL_Delay(1);
  }

  bme280_read_calibration(hi2c);

  /* Default to a low-power "weather monitoring" style setup:
     forced mode, osrs_h=1, osrs_t=1, osrs_p=1, filter off.
     (You can trigger measurement when you want.) */
  bme280_config_weather_forced(hi2c);
}

void bme280_config_weather_forced(I2C_HandleTypeDef *hi2c)
{
  /* config: t_sb ignored in forced, filter off */
  uint8_t config = 0;
  config |= (BME280_FILTER_OFF << 2);     // filter[2:0]
  config |= (BME280_TSB_1000MS << 5);     // t_sb[2:0] (doesn't matter in forced)
  bme280_write_u8(hi2c,BME280_REG_CONFIG, config);

  /* ctrl_hum must be written, then ctrl_meas to make it effective */
  bme280_write_u8(hi2c,BME280_REG_CTRL_HUM, BME280_OSRS_X1); // osrs_h = x1

  /* ctrl_meas: osrs_t=x1, osrs_p=x1, mode=sleep for now */
  uint8_t ctrl_meas = 0;
  ctrl_meas |= (BME280_OSRS_X1 << 5);     // osrs_t
  ctrl_meas |= (BME280_OSRS_X1 << 2);     // osrs_p
  ctrl_meas |= BME280_MODE_SLEEP;         // mode
  bme280_write_u8(hi2c,BME280_REG_CTRL_MEAS, ctrl_meas);
}

void bme280_trigger_forced(I2C_HandleTypeDef *hi2c)
{
  /* Read current ctrl_meas and only change mode bits */
  uint8_t ctrl_meas = bme280_read_u8(hi2c,BME280_REG_CTRL_MEAS);
  ctrl_meas = (uint8_t)((ctrl_meas & ~0x03) | BME280_MODE_FORCED);
  bme280_write_u8(hi2c,BME280_REG_CTRL_MEAS, ctrl_meas);
}

bool bme280_is_measuring(I2C_HandleTypeDef *hi2c)
{
  uint8_t st = bme280_read_u8(hi2c,BME280_REG_STATUS);
  return ((st >> 3) & 0x01) ? true : false; // measuring bit
}

bool bme280_read_compensated(I2C_HandleTypeDef *hi2c,bme280_data_t *out)
{
  if (!out || !s_cal.calib_loaded) return false;

  /* Burst read 0xF7..0xFE (press(3)+temp(3)+hum(2)) */
  uint8_t buf[8] = {0};
  bme280_i2c_read(hi2c,BME280_REG_PRESS_MSB, buf, sizeof(buf));

  int32_t adc_P = (int32_t)((((uint32_t)buf[0] << 12) | ((uint32_t)buf[1] << 4) | ((uint32_t)buf[2] >> 4)));
  int32_t adc_T = (int32_t)((((uint32_t)buf[3] << 12) | ((uint32_t)buf[4] << 4) | ((uint32_t)buf[5] >> 4)));
  int32_t adc_H = (int32_t)((((uint32_t)buf[6] << 8)  | ((uint32_t)buf[7])));

  /* Compensate */
  int32_t  T_01C = bme280_compensate_T_int32(hi2c,adc_T);     // 0.01 °C
  uint32_t P_Q24_8 = bme280_compensate_P_int64(hi2c,adc_P);   // Pa in Q24.8
  uint32_t H_Q22_10 = bme280_compensate_H_int32(hi2c,adc_H);  // %RH in Q22.10

  out->temperature_c = (float)T_01C / 100.0f;
  out->pressure_pa   = (float)P_Q24_8 / 256.0f;
  out->humidity_rh   = (float)H_Q22_10 / 1024.0f;

  return true;
}

bool bme280_read_forced(I2C_HandleTypeDef *hi2c, bme280_data_t *out)
{
	  bme280_trigger_forced(hi2c);
	  HAL_Delay(10);
	  uint32_t t0 = HAL_GetTick();
	  while (bme280_is_measuring(hi2c))
	  {
	    if ((HAL_GetTick() - t0) > 200) return false;
	  }

	  return bme280_read_compensated(hi2c, out);
}

/* Optional compatibility placeholders */
bool bme280_open(I2C_HandleTypeDef *hi2c)   { return true; }
bool bme280_update(I2C_HandleTypeDef *hi2c) { return true; }
