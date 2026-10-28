/**
 ******************************************************************************
 * @file    tmp117.c
 * @author  Daisy Berrios
 * @date    Jan 13, 2026
 * @brief   Driver for TMP117 high-accuracy temperature sensor
 *******************************************************************************
 */

#include "tmp117.h"
#include "main.h"


/*- PRIVATE_Definitions ------------------------------------------------------*/
#define TIMEOUT_100MS   100U
#define TMP117_ID_EXPECTED  (0x0117U)
#define TMP117_RESOLUTION_C 0.0078125f

/*- PRIVATE_Data -------------------------------------------------------------*/
static uint16_t g_tmp117_i2c_addr = TMP117_I2C_ADDR_DEFAULT;

/*- PRIVATE_Functions --------------------------------------------------------*/
static HAL_StatusTypeDef tmp117_read_reg16(I2C_HandleTypeDef *hi2c,uint8_t reg, uint16_t *val)
{
  uint8_t buf[2];
  HAL_StatusTypeDef st;

  if (val == NULL) return HAL_ERROR;

  st = HAL_I2C_Mem_Read(hi2c, g_tmp117_i2c_addr, reg, 1, buf, 2, TIMEOUT_100MS);
  if (st != HAL_OK) return st;

  *val = (uint16_t)((buf[0] << 8) | buf[1]); // MSB first
  return HAL_OK;
}

static HAL_StatusTypeDef tmp117_write_reg16(I2C_HandleTypeDef *hi2c,uint8_t reg, uint16_t val)
{
  uint8_t buf[2];
  buf[0] = (uint8_t)(val >> 8);
  buf[1] = (uint8_t)(val & 0xFF);

  return HAL_I2C_Mem_Write(hi2c, g_tmp117_i2c_addr, reg, 1, buf, 2, TIMEOUT_100MS);
}

static HAL_StatusTypeDef tmp117_update_bits(I2C_HandleTypeDef *hi2c,uint8_t reg, uint16_t mask, uint16_t value)
{
  uint16_t cur = 0;
  HAL_StatusTypeDef st = tmp117_read_reg16(hi2c,reg, &cur);
  if (st != HAL_OK) return st;

  cur = (uint16_t)((cur & ~mask) | (value & mask));
  return tmp117_write_reg16(hi2c,reg, cur);
}

static float tmp117_raw_to_c(int16_t raw)
{
  return ((float)raw) * TMP117_RESOLUTION_C;
}

/*- PUBLIC_API ---------------------------------------------------------------*/

void tmp117_set_i2c_address(I2C_HandleTypeDef *hi2c,tmp117_addr_t addr_sel)
{
  // 7-bit: 0x48..0x4B (ADD0 = GND, V+, SDA, SCL)
  // HAL expects left-shifted address.
  switch (addr_sel)
  {
    case TMP117_ADDR_GND: g_tmp117_i2c_addr = (0x48U << 1); break;
    case TMP117_ADDR_VDD: g_tmp117_i2c_addr = (0x49U << 1); break;
    case TMP117_ADDR_SDA: g_tmp117_i2c_addr = (0x4AU << 1); break;
    case TMP117_ADDR_SCL: g_tmp117_i2c_addr = (0x4BU << 1); break;
    default:              g_tmp117_i2c_addr = (0x48U << 1); break;
  }
}

bool tmp117_init(I2C_HandleTypeDef *hi2c)
{
  uint16_t id = 0;

  // Read Device ID (0x0F). Expected default = 0x0117.
  if (tmp117_read_reg16(hi2c,TMP117_REG_DEVICE_ID, &id) != HAL_OK)
  {
        return false;
  }

  if (id != TMP117_ID_EXPECTED)
  {
    return false;
  }

  return true;
}

bool tmp117_soft_reset(I2C_HandleTypeDef *hi2c)
{
  // Set Soft_Reset bit (bit1). 
  if (tmp117_update_bits(hi2c,TMP117_REG_CONFIG, TMP117_CFG_SOFT_RESET, TMP117_CFG_SOFT_RESET) != HAL_OK)
  {
    Error_Handler();
    return false;
  }
  return true;
}

bool tmp117_set_mode(I2C_HandleTypeDef *hi2c,tmp117_mode_t mode)
{
  uint16_t v = 0;
  switch (mode)
  {
    case TMP117_MODE_CONTINUOUS: v = (0U << 10); break; // MOD[1:0]=00
    case TMP117_MODE_SHUTDOWN:   v = (1U << 10); break; // MOD[1:0]=01
    case TMP117_MODE_ONESHOT:    v = (3U << 10); break; // MOD[1:0]=11
    default: return false;
  }

  if (tmp117_update_bits(hi2c,TMP117_REG_CONFIG, TMP117_CFG_MOD_MASK, v) != HAL_OK)
  {
    Error_Handler();
    return false;
  }
  return true;
}

bool tmp117_set_averaging(I2C_HandleTypeDef *hi2c,tmp117_avg_t avg)
{
  uint16_t v = 0;
  switch (avg)
  {
    case TMP117_AVG_1:  v = (0U << 5); break; // AVG[1:0]=00
    case TMP117_AVG_8:  v = (1U << 5); break; // 01
    case TMP117_AVG_32: v = (2U << 5); break; // 10
    case TMP117_AVG_64: v = (3U << 5); break; // 11
    default: return false;
  }

  if (tmp117_update_bits(hi2c,TMP117_REG_CONFIG, TMP117_CFG_AVG_MASK, v) != HAL_OK)
  {
    Error_Handler();
    return false;
  }
  return true;
}

bool tmp117_set_conv_time_cc(I2C_HandleTypeDef *hi2c,tmp117_conv_t conv)
{
  uint16_t v = ((uint16_t)conv & 0x7U) << 7;

  if (tmp117_update_bits(hi2c,TMP117_REG_CONFIG, TMP117_CFG_CONV_MASK, v) != HAL_OK)
  {
    Error_Handler();
    return false;
  }
  return true;
}

bool tmp117_read_temperature_c(I2C_HandleTypeDef *hi2c,float *temp_c)
{
  uint16_t raw_u = 0;
  int16_t raw_s;

  if (temp_c == NULL) return false;

  if (tmp117_read_reg16(hi2c,TMP117_REG_TEMP_RESULT, &raw_u) != HAL_OK)
  {
    Error_Handler();
    return false;
  }

  raw_s = (int16_t)raw_u;
  *temp_c = tmp117_raw_to_c(raw_s);
  return true;
}

bool tmp117_read_temperature_mC(I2C_HandleTypeDef *hi2c,int32_t *temp_mC)
{
  float tC;
  if (temp_mC == NULL) return false;

  if (!tmp117_read_temperature_c(hi2c,&tC)) return false;

  *temp_mC = (int32_t)(tC * 1000.0f + (tC >= 0 ? 0.5f : -0.5f));
  return true;
}

bool tmp117_read_config(I2C_HandleTypeDef *hi2c,uint16_t *cfg)
{
  if (cfg == NULL) return false;

  if (tmp117_read_reg16(hi2c,TMP117_REG_CONFIG, cfg) != HAL_OK)
  {
    Error_Handler();
    return false;
  }
  return true;
}

bool tmp117_wait_data_ready(I2C_HandleTypeDef *hi2c,uint32_t timeout_ms)
{
  uint32_t t0 = HAL_GetTick();

  while ((HAL_GetTick() - t0) < timeout_ms)
  {
    uint16_t cfg = 0;
    if (!tmp117_read_config(hi2c,&cfg)) return false;

    if (cfg & TMP117_CFG_DATA_READY)
    {
      return true;
    }
  }
  return false;
}

bool tmp117_oneshot_read_temperature_c(I2C_HandleTypeDef *hi2c,float *temp_c, uint32_t timeout_ms)
{
  // Trigger one-shot conversion, wait for Data_Ready, then read temperature.
  if (!tmp117_set_mode(hi2c,TMP117_MODE_ONESHOT)) return false;
  HAL_Delay(20);
  if (!tmp117_wait_data_ready(hi2c,timeout_ms)) return false;

  return tmp117_read_temperature_c(hi2c,temp_c);
}

bool tmp117_set_limits_c(I2C_HandleTypeDef *hi2c,float t_low_c, float t_high_c)
{
  int32_t low_raw  = (int32_t)(t_low_c  / TMP117_RESOLUTION_C);
  int32_t high_raw = (int32_t)(t_high_c / TMP117_RESOLUTION_C);

  if (low_raw < -32768) low_raw = -32768;
  if (low_raw >  32767) low_raw =  32767;
  if (high_raw < -32768) high_raw = -32768;
  if (high_raw >  32767) high_raw =  32767;

  if (tmp117_write_reg16(hi2c,TMP117_REG_TLOW,  (uint16_t)((int16_t)low_raw)) != HAL_OK)
  {
    Error_Handler();
    return false;
  }
  if (tmp117_write_reg16(hi2c,TMP117_REG_THIGH, (uint16_t)((int16_t)high_raw)) != HAL_OK)
  {
    Error_Handler();
    return false;
  }
  return true;
}
