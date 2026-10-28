/**
 ******************************************************************************
 * @file    rv3028.c
 * @author  Daisy Berrios
 * @date    Jan 13, 2026
 * @brief   Driver for Micro Crystal RV-3028-C7 RTC
 *
 * @note    I2C 7-bit address is 0x52 (1010010b). Use (0x52<<1) with STM32 HAL.
 * @warning Requires STM32 HAL I2C (i2c.h) and Error_Handler().
 *******************************************************************************
 */

#include "rv3028.h"
#include <string.h>

/*- PRIVATE_Definitions ------------------------------------------------------*/
#define TIMEOUT_100MS 100

/*- PRIVATE_Functions --------------------------------------------------------*/
static uint8_t bcd_to_bin(uint8_t bcd) {
  return (uint8_t)(((bcd >> 4) * 10U) + (bcd & 0x0FU));
}

static uint8_t bin_to_bcd(uint8_t bin) {
  return (uint8_t)(((bin / 10U) << 4) | (bin % 10U));
}

static HAL_StatusTypeDef rv3028_read(rv3028_t *dev, uint8_t reg, uint8_t *data, uint16_t len) {
  return HAL_I2C_Mem_Read(dev->i2c_handle, dev->dev_addr, reg, 1, data, len, TIMEOUT_100MS);
}

static HAL_StatusTypeDef rv3028_write(rv3028_t *dev, uint8_t reg, const uint8_t *data, uint16_t len) {
  return HAL_I2C_Mem_Write(dev->i2c_handle, dev->dev_addr, reg, 1, (uint8_t*)data, len, TIMEOUT_100MS);
}

/*- PUBLIC_API ---------------------------------------------------------------*/

bool rv3028_init(rv3028_t *dev, I2C_HandleTypeDef *i2c_handle, uint8_t addr) {
    if (!dev || !i2c_handle) return false;
    dev->i2c_handle = i2c_handle;
    dev->dev_addr = addr;

    uint8_t id = 0;
    // Attempt multiple reads if the first one fails due to noise
    for(int retry = 0; retry < 3; retry++) {
        if (rv3028_read(dev, RV3028_REG_ID, &id, 1) == HAL_OK) break;
        HAL_Delay(10);
        if (retry == 2) return false;
    }

    /* --- EEPROM Power Configuration --- */
    uint8_t ctrl1, backup_reg;

    // Disable Refresh (EERD)
    rv3028_read(dev, RV3028_REG_CONTROL1, &ctrl1, 1);
    ctrl1 |= (1U << 3);
    rv3028_write(dev, RV3028_REG_CONTROL1, &ctrl1, 1);

    // Set BSM = 11 (Level Switching Mode) + FEDE = 1 (Fast Edge Detection)
    // Register 0x37: [X][X][X][FEDE][BSM1][BSM0][X][X]
    rv3028_read(dev, RV3028_REG_BACKUP, &backup_reg, 1);
    backup_reg &= ~(0x0C);
    backup_reg |= 0x1C; // FEDE=1, BSM=11
    rv3028_write(dev, RV3028_REG_BACKUP, &backup_reg, 1);

    // Commit RAM to EEPROM
    uint8_t update_cmd = 0x11;
    rv3028_write(dev, RV3028_REG_EECMD, &update_cmd, 1);

    // Wait for EEbusy (Register 0x0E, Bit 7)
    uint8_t status = 0;
    do {
        rv3028_read(dev, RV3028_REG_STATUS, &status, 1);
    } while (status & (1U << 7));

    // Re-enable Refresh
    ctrl1 &= ~(1U << 3);
    rv3028_write(dev, RV3028_REG_CONTROL1, &ctrl1, 1);

    rv3028_set_24h_mode(dev, true);
    rv3028_clear_flags(dev, RV3028_FLAG_PORF);

    return true;
}

bool rv3028_wakeup_init(rv3028_t *dev) {
    if (!dev || !dev->i2c_handle) return false;

    // 1. Clear any I2C errors that might have occurred during power-off/on
    __HAL_I2C_CLEAR_FLAG(dev->i2c_handle, I2C_FLAG_AF | I2C_FLAG_BERR | I2C_FLAG_ARLO);

    // 2. Small delay to let the RTC internal I2C logic stabilize after V_SENSOR returns
    HAL_Delay(5);

    // 3. Verify communication by reading the Status Register
    uint8_t status = 0;
    if (!rv3028_get_flags(dev, &status)) {
        return false;
    }

    // 4. Clear the specific flag that woke us up (Timer or Alarm)
    // Writing 0 to the flag bits in the Status register clears them.
    uint8_t clear_mask = 0;
    if (status & RV3028_FLAG_TF) clear_mask |= RV3028_FLAG_TF;
    if (status & RV3028_FLAG_AF) clear_mask |= RV3028_FLAG_AF;

    if (clear_mask > 0) {
        // Use your existing clear_flags function
        if (!rv3028_clear_flags(dev, clear_mask)) return false;
    }

    return true;
}

bool rv3028_set_24h_mode(rv3028_t *dev, bool enable_24h) {
  uint8_t c2 = 0;
  if (rv3028_read(dev, RV3028_REG_CONTROL2, &c2, 1) != HAL_OK) return false;

  if (enable_24h) c2 &= (uint8_t)~RV3028_CTRL2_12_24;
  else            c2 |= RV3028_CTRL2_12_24;

  return (rv3028_write(dev, RV3028_REG_CONTROL2, &c2, 1) == HAL_OK);
}

bool rv3028_get_datetime(rv3028_t *dev, rv3028_datetime_t *dt) {
  uint8_t buf[7];
  if (!dev || !dt) return false;

  if (rv3028_read(dev, RV3028_REG_SECONDS, buf, sizeof(buf)) != HAL_OK) return false;

  dt->seconds = bcd_to_bin(buf[0] & 0x7F);
  dt->minutes = bcd_to_bin(buf[1] & 0x7F);
  dt->hours   = bcd_to_bin(buf[2] & 0x3F);
  dt->weekday = (uint8_t)(buf[3] & 0x07);
  dt->date    = bcd_to_bin(buf[4] & 0x3F);
  dt->month   = bcd_to_bin(buf[5] & 0x1F);
  dt->year    = (uint16_t)(2000U + bcd_to_bin(buf[6]));

  return true;
}

bool rv3028_set_datetime(rv3028_t *dev, const rv3028_datetime_t *dt) {
  uint8_t buf[7];
  if (!dev || !dt) return false;

  buf[0] = bin_to_bcd(dt->seconds);
  buf[1] = bin_to_bcd(dt->minutes);
  buf[2] = bin_to_bcd(dt->hours);
  buf[3] = (uint8_t)(dt->weekday & 0x07);
  buf[4] = bin_to_bcd(dt->date);
  buf[5] = bin_to_bcd(dt->month);
  buf[6] = bin_to_bcd((uint8_t)(dt->year - 2000U));

  return (rv3028_write(dev, RV3028_REG_SECONDS, buf, sizeof(buf)) == HAL_OK);
}

bool rv3028_alarm_set_daily(rv3028_t *dev, uint8_t hour, uint8_t minute) {
  if (minute > 59 || hour > 23) return false;

  /* 1. Disable current alarm interrupts to prevent false triggers */
  uint8_t c2;
  rv3028_read(dev, RV3028_REG_CONTROL2, &c2, 1);
  c2 &= ~RV3028_CTRL2_AIE;
  rv3028_write(dev, RV3028_REG_CONTROL2, &c2, 1);

  /* 2. Set Alarm Registers (AE_M=0, AE_H=0, AE_WD=1) */
  uint8_t a[3];
  a[0] = (uint8_t)(0U << 7) | (bin_to_bcd(minute) & 0x7F);
  a[1] = (uint8_t)(0U << 7) | (bin_to_bcd(hour)   & 0x3F);
  a[2] = (uint8_t)(1U << 7); /* Disable Weekday/Date match */

  if (rv3028_write(dev, RV3028_REG_MIN_ALARM, a, sizeof(a)) != HAL_OK) return false;

  /* 3. Clear Flag and Enable Interrupt */
  rv3028_clear_flags(dev, RV3028_FLAG_AF);
  c2 |= RV3028_CTRL2_AIE;

  return (rv3028_write(dev, RV3028_REG_CONTROL2, &c2, 1) == HAL_OK);
}

bool rv3028_alarm_disable(rv3028_t *dev) {
  uint8_t c2;
  if (rv3028_read(dev, RV3028_REG_CONTROL2, &c2, 1) != HAL_OK) return false;
  c2 &= ~RV3028_CTRL2_AIE;
  return (rv3028_write(dev, RV3028_REG_CONTROL2, &c2, 1) == HAL_OK);
}

bool rv3028_timer_set(rv3028_t *dev, uint16_t value, uint8_t freq, bool repeat) {
  /* * freq map: 0=4096Hz, 1=64Hz, 2=1Hz, 3=60s
   * value: 12-bit countdown value (0 to 4095)
   */
  if (freq > 3 || value > 4095) return false;

  /* 1. Disable Timer (TE=0) to configure safely */
  uint8_t c1;
  rv3028_read(dev, RV3028_REG_CONTROL1, &c1, 1);
  c1 &= ~RV3028_CTRL1_TE;
  rv3028_write(dev, RV3028_REG_CONTROL1, &c1, 1);

  /* 2. Set Timer Value (Low byte, High byte) */
  uint8_t t_vals[2];
  t_vals[0] = (uint8_t)(value & 0xFF);
  t_vals[1] = (uint8_t)((value >> 8) & 0x0F);
  if (rv3028_write(dev, RV3028_REG_TIMER_VAL0, t_vals, 2) != HAL_OK) return false;

  /* 3. Configure Control1 (Clock Freq, Repeat, Enable) */
  c1 &= ~0x03; // Clear old freq bits
  c1 |= (freq & 0x03);

  if (repeat) c1 |= RV3028_CTRL1_TRPT;
  else        c1 &= ~RV3028_CTRL1_TRPT;

  c1 |= RV3028_CTRL1_TE; // Enable Timer

  if (rv3028_write(dev, RV3028_REG_CONTROL1, &c1, 1) != HAL_OK) return false;

  /* 4. Enable Timer Interrupt (TIE) in Control2 */
  uint8_t c2;
  rv3028_read(dev, RV3028_REG_CONTROL2, &c2, 1);
  c2 |= RV3028_CTRL2_TIE;

  // Clear any old Timer Flag before enabling
  rv3028_clear_flags(dev, RV3028_FLAG_TF);

  return (rv3028_write(dev, RV3028_REG_CONTROL2, &c2, 1) == HAL_OK);
}

bool rv3028_timer_disable(rv3028_t *dev) {
  uint8_t c1, c2;

  // Disable Timer Enable bit
  if (rv3028_read(dev, RV3028_REG_CONTROL1, &c1, 1) != HAL_OK) return false;
  c1 &= ~RV3028_CTRL1_TE;
  rv3028_write(dev, RV3028_REG_CONTROL1, &c1, 1);

  // Disable Timer Interrupt bit
  if (rv3028_read(dev, RV3028_REG_CONTROL2, &c2, 1) != HAL_OK) return false;
  c2 &= ~RV3028_CTRL2_TIE;
  return (rv3028_write(dev, RV3028_REG_CONTROL2, &c2, 1) == HAL_OK);
}

bool rv3028_clear_flags(rv3028_t *dev, uint8_t flags_mask) {
  uint8_t st = 0;
  if (rv3028_read(dev, RV3028_REG_STATUS, &st, 1) != HAL_OK) return false;

  st &= ~flags_mask; /* Clear specific flags by writing 0 */

  return (rv3028_write(dev, RV3028_REG_STATUS, &st, 1) == HAL_OK);
}

bool rv3028_get_flags(rv3028_t *dev, uint8_t *flags) {
  return (rv3028_read(dev, RV3028_REG_STATUS, flags, 1) == HAL_OK);
}

void rv3028_handle_irq(rv3028_t *dev) {
  uint8_t flags = 0;
  if (rv3028_get_flags(dev, &flags)) {
    // If Alarm Flag is set, clear it
    if (flags & RV3028_FLAG_AF) {
       rv3028_clear_flags(dev, RV3028_FLAG_AF);
    }
    // If Timer Flag is set, clear it
    if (flags & RV3028_FLAG_TF) {
       rv3028_clear_flags(dev, RV3028_FLAG_TF);
    }
  }
}
