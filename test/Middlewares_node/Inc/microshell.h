/**
 ******************************************************************************
 * @file    microshell.h
 * @author  Daisy Berrios
 * @date    Jan 13, 2026
 * @brief   Minimal USB-CDC microshell (commands over USB_OTG_FS)
 *
 * @note    Feed RX bytes from CDC_Receive callback.
 * @note    Call microshell_task() periodically in main loop to flush TX queue.
 ******************************************************************************
 */
#ifndef _INC_MICROSHELL_H_
#define _INC_MICROSHELL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>

/*- USER_Definitions --------------------------------------------------------*/
#ifndef MICROSHELL_RX_BUF_SZ
#define MICROSHELL_RX_BUF_SZ   256
#endif

#ifndef MICROSHELL_TX_BUF_SZ
#define MICROSHELL_TX_BUF_SZ   512
#endif

#ifndef MICROSHELL_MAX_TOKENS
#define MICROSHELL_MAX_TOKENS  8
#endif

/*- PUBLIC_Types ------------------------------------------------------------*/
typedef struct
{
  uint16_t year;   /* e.g. 2026 */
  uint8_t  month;  /* 1..12 */
  uint8_t  day;    /* 1..31 */
  uint8_t  hour;   /* 0..23 */
  uint8_t  min;    /* 0..59 */
  uint8_t  sec;    /* 0..59 */
} microshell_datetime_t;

/* Application hooks (you implement these in your app, then register) */
typedef struct
{
  /* RTC */
  bool (*rtc_get)(microshell_datetime_t *out);
  bool (*rtc_set)(const microshell_datetime_t *dt);

  /* Intervals (seconds) */
  bool (*wakeup_get_s)(uint32_t *out_s);
  bool (*wakeup_set_s)(uint32_t s);

  bool (*txint_get_s)(uint32_t *out_s);
  bool (*txint_set_s)(uint32_t s);

  /* Data retrieval (optional) */
  bool (*data_list)(void);                 /* prints list via microshell_print* */
  bool (*data_read)(uint32_t index);       /* prints record/index via print*   */
  bool (*data_clear)(void);                /* clears stored data               */

  /* Optional: called on every unknown command line */
  bool (*custom_line)(int argc, char **argv);
} microshell_hooks_t;

/*- PUBLIC_API --------------------------------------------------------------*/
void microshell_init(const microshell_hooks_t *hooks);

/* Feed raw RX bytes from USB CDC receive */
void microshell_rx_bytes(const uint8_t *data, uint16_t len);

/* Call periodically to flush TX queue (and send prompt) */
void microshell_task(void);

/* Printing helpers (go to USB) */
void microshell_print(const char *s);
void microshell_printf(const char *fmt, ...);

/* Optional: if you want to force a prompt */
void microshell_prompt(void);

#ifdef __cplusplus
}
#endif

#endif /* _INC_MICROSHELL_H_ */
