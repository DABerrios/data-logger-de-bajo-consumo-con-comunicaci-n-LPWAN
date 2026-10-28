/**
 ******************************************************************************
 * @file    rv3028.h
 * @author  ELO301
 * @date    Jan 13, 2026
 * @brief   Driver for Micro Crystal RV-3028-C7 RTC
 *
 * <h2><center>&copy; COPYRIGHT name</center></h2>
 *******************************************************************************
 */
#ifndef _INC_RV3028_H_
#define _INC_RV3028_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include "main.h" /* Ensure this includes stm32l4xx_hal.h */

/*- USER_Definitions ------------------------------------------------------*/
#define RV3028_I2C_ADDR_DEFAULT      (0x52U << 1)

/* Register map */
#define RV3028_REG_SECONDS           0x00
#define RV3028_REG_MINUTES           0x01
#define RV3028_REG_HOURS             0x02
#define RV3028_REG_WEEKDAY           0x03
#define RV3028_REG_DATE              0x04
#define RV3028_REG_MONTH             0x05
#define RV3028_REG_YEAR              0x06
#define RV3028_REG_MIN_ALARM         0x07
#define RV3028_REG_HOUR_ALARM        0x08
#define RV3028_REG_WD_DATE_ALARM     0x09
#define RV3028_REG_TIMER_VAL0        0x0A
#define RV3028_REG_TIMER_VAL1        0x0B
#define RV3028_REG_TIMER_STAT0       0x0C
#define RV3028_REG_TIMER_STAT1       0x0D
#define RV3028_REG_STATUS            0x0E
#define RV3028_REG_CONTROL1          0x0F
#define RV3028_REG_CONTROL2          0x10
#define RV3028_REG_EECMD             0x27  /* EEPROM Command Register */
#define RV3028_REG_ID                0x28
#define RV3028_REG_BACKUP            0x37

#define RV3028_BACKUP_BSM_LSM        (0x0C)    /* Level Switching Mode (11) */
#define RV3028_BACKUP_FEDE           (1U << 4)

/* STATUS bits (0Eh) */
#define RV3028_STATUS_EEBUSY         (1U << 7) /* EEPROM Busy bit */
// ... existing flags ...

/* CONTROL1 bits (0Fh) */
#define RV3028_CTRL1_EERD            (1U << 3) /* EEPROM Memory Refresh Disable */
#define RV3028_CTRL1_TE              (1U << 2)

/* STATUS bits (0Eh) */
#define RV3028_FLAG_CLKF             (1U << 6)
#define RV3028_FLAG_BSF              (1U << 5)
#define RV3028_FLAG_UF               (1U << 4)
#define RV3028_FLAG_TF               (1U << 3)
#define RV3028_FLAG_AF               (1U << 2)
#define RV3028_FLAG_EVF              (1U << 1)
#define RV3028_FLAG_PORF             (1U << 0)

/* CONTROL1 bits (0Fh) */
#define RV3028_CTRL1_TRPT            (1U << 7)  /* Timer Repeat */
#define RV3028_CTRL1_TE              (1U << 2)  /* Timer Enable */
#define RV3028_CTRL1_TD_4096HZ       (0x00)     /* Timer Clock Frequencies */
#define RV3028_CTRL1_TD_64HZ         (0x01)
#define RV3028_CTRL1_TD_1HZ          (0x02)
#define RV3028_CTRL1_TD_60S          (0x03)

/* CONTROL2 bits (10h) */
#define RV3028_CTRL2_TIE             (1U << 4)  /* Timer Interrupt Enable */
#define RV3028_CTRL2_AIE             (1U << 3)  /* Alarm Interrupt Enable */
#define RV3028_CTRL2_12_24           (1U << 1)

/*- PUBLIC_Types ------------------------------------------------------------*/

typedef struct {
  I2C_HandleTypeDef *i2c_handle;
  uint8_t dev_addr;
} rv3028_t;

typedef struct {
  uint16_t year;   /* 2000..2099 */
  uint8_t  month;  /* 1..12 */
  uint8_t  date;   /* 1..31 */
  uint8_t  weekday;/* 0..6 */
  uint8_t  hours;  /* 0..23 */
  uint8_t  minutes;/* 0..59 */
  uint8_t  seconds;/* 0..59 */
} rv3028_datetime_t;

/*- PUBLIC_API ---------------------------------------------------------------*/

/* Initialization */
bool rv3028_init(rv3028_t *dev, I2C_HandleTypeDef *i2c_handle, uint8_t addr);

/* Core RTC Functions */
bool rv3028_get_datetime(rv3028_t *dev, rv3028_datetime_t *dt);
bool rv3028_set_datetime(rv3028_t *dev, const rv3028_datetime_t *dt);
bool rv3028_set_24h_mode(rv3028_t *dev, bool enable_24h);

/* Alarm Functions (Daily Alarm) */
bool rv3028_alarm_set_daily(rv3028_t *dev, uint8_t hour, uint8_t minute);
bool rv3028_alarm_disable(rv3028_t *dev);

/* Countdown Timer Functions (For Wakeup) */
/* freq: 0=4096Hz, 1=64Hz, 2=1Hz, 3=1/60Hz */
bool rv3028_timer_set(rv3028_t *dev, uint16_t value, uint8_t freq, bool repeat);
bool rv3028_timer_disable(rv3028_t *dev);

/* Interrupt Flag Handling */
bool rv3028_clear_flags(rv3028_t *dev, uint8_t flags_mask);
bool rv3028_get_flags(rv3028_t *dev, uint8_t *flags);

/* Call this after waking up from STOP mode to clear INT pin */
void rv3028_handle_irq(rv3028_t *dev);
bool rv3028_wakeup_init(rv3028_t *dev);
#ifdef __cplusplus
}
#endif

#endif /* _INC_RV3028_H_ */
