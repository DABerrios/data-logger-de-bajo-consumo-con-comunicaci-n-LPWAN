/**
 ******************************************************************************
 * @file    tmp117.h
 * @author  Daisy Berrios
 * @date    Jan 13, 2026
 * @brief   Driver for TMP117 high-accuracy temperature sensor
 *******************************************************************************
 */
#ifndef _INC_TMP117_H_
#define _INC_TMP117_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include "main.h"

/*- Register Map ------------------------------------------------------------*/
#define TMP117_REG_TEMP_RESULT   0x00
#define TMP117_REG_CONFIG        0x01
#define TMP117_REG_THIGH         0x02
#define TMP117_REG_TLOW          0x03
#define TMP117_REG_EEPROM_UL     0x04
#define TMP117_REG_EEPROM1       0x05
#define TMP117_REG_EEPROM2       0x06
#define TMP117_REG_TEMP_OFFSET   0x07
#define TMP117_REG_EEPROM3       0x08
#define TMP117_REG_DEVICE_ID     0x0F

/*- I2C address (ADD0 = V+) ----------------------------------------*/
#define TMP117_I2C_ADDR_DEFAULT  (0x49U << 1)

/*- Configuration Register Bits (0x01) --------------------------------------*/
#define TMP117_CFG_HIGH_ALERT    (1U << 15)
#define TMP117_CFG_LOW_ALERT     (1U << 14)
#define TMP117_CFG_DATA_READY    (1U << 13)
#define TMP117_CFG_EEPROM_BUSY   (1U << 12)

#define TMP117_CFG_MOD_MASK      (3U << 10)  // MOD[1:0] bits 11:10
#define TMP117_CFG_CONV_MASK     (7U << 7)   // CONV[2:0] bits 9:7
#define TMP117_CFG_AVG_MASK      (3U << 5)   // AVG[1:0] bits 6:5

#define TMP117_CFG_TnA           (1U << 4)
#define TMP117_CFG_POL           (1U << 3)
#define TMP117_CFG_DR_ALERT      (1U << 2)
#define TMP117_CFG_SOFT_RESET    (1U << 1)

/*- Types -------------------------------------------------------------------*/
typedef enum
{
  TMP117_ADDR_GND = 0,
  TMP117_ADDR_VDD = 1,
  TMP117_ADDR_SDA = 2,
  TMP117_ADDR_SCL = 3
} tmp117_addr_t;

typedef enum
{
  TMP117_MODE_CONTINUOUS = 0,
  TMP117_MODE_SHUTDOWN   = 1,
  TMP117_MODE_ONESHOT    = 2
} tmp117_mode_t;

typedef enum
{
  TMP117_AVG_1  = 0,
  TMP117_AVG_8  = 1,
  TMP117_AVG_32 = 2,
  TMP117_AVG_64 = 3
} tmp117_avg_t;

typedef enum
{
  // CONV[2:0] in CC mode:
  // 000=15.5ms, 001=125ms, 010=250ms, 011=500ms, 100=1s, 101=4s, 110=8s, 111=16s
  TMP117_CONV_15p5MS = 0,
  TMP117_CONV_125MS  = 1,
  TMP117_CONV_250MS  = 2,
  TMP117_CONV_500MS  = 3,
  TMP117_CONV_1S     = 4,
  TMP117_CONV_4S     = 5,
  TMP117_CONV_8S     = 6,
  TMP117_CONV_16S    = 7
} tmp117_conv_t;

/*- Public API ---------------------------------------------------------------*/
void tmp117_set_i2c_address(I2C_HandleTypeDef *hi2c,tmp117_addr_t addr_sel);

bool tmp117_init(I2C_HandleTypeDef *hi2c);
bool tmp117_soft_reset(I2C_HandleTypeDef *hi2c);

bool tmp117_set_mode(I2C_HandleTypeDef *hi2c,tmp117_mode_t mode);
bool tmp117_set_averaging(I2C_HandleTypeDef *hi2c,tmp117_avg_t avg);
bool tmp117_set_conv_time_cc(I2C_HandleTypeDef *hi2c,tmp117_conv_t conv);

bool tmp117_read_temperature_c(I2C_HandleTypeDef *hi2c,float *temp_c);
bool tmp117_read_temperature_mC(I2C_HandleTypeDef *hi2c,int32_t *temp_mC);

bool tmp117_read_config(I2C_HandleTypeDef *hi2c,uint16_t *cfg);
bool tmp117_wait_data_ready(I2C_HandleTypeDef *hi2c,uint32_t timeout_ms);

/* Convenience: trigger one-shot, wait DR, read temp */
bool tmp117_oneshot_read_temperature_c(I2C_HandleTypeDef *hi2c,float *temp_c, uint32_t timeout_ms);

/* Optional: set THIGH/TLOW in °C (same format as temp register) */
bool tmp117_set_limits_c(I2C_HandleTypeDef *hi2c,float t_low_c, float t_high_c);

#ifdef __cplusplus
}
#endif

#endif /* _INC_TMP117_H_ */
