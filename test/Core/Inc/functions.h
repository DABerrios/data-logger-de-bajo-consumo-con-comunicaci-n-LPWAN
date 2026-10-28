/**
 ******************************************************************************
 * @file    functions.h
 * @author  Daisy Berrios
 * @date    Jan 13, 2026
 * @brief   Public API for sensor logging, LoRaWAN uplink, and power control.
 *******************************************************************************
 */
#ifndef INC_FUNCTIONS_H_
#define INC_FUNCTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include <stdint.h>
#include "rv3028.h"

#include "LoRaMac.h"
#include "LmHandler.h"
#include "Commissioning.h"

#define ACTIVE_REGION       LORAMAC_REGION_AU915
#define TX_INTERVAL_MS      10000
#define LORAWAN_APP_PORT    2

/*
 * 7-byte LoRaWAN uplink payload — big-endian, no timestamp (TTN provides it).
 *
 * Offset  Bytes  Field       Encoding
 * ------  -----  ----------  ------------------------------------------
 *  0-1      2    rain        uint16 BE  cumulative pulses (24 h reset)
 *  2-3      2    temp        int16  BE  degC x 100   (averaged)
 *  4        1    humidity    uint8      %             (averaged)
 *  5-6      2    pressure    uint16 BE  hPa x 10     (averaged)
 */
#define LORA_PAYLOAD_SIZE  7

/* Sensor averaging state accumulated between LoRa uplinks. */
typedef struct {
    float   sum_temp;       /* degC accumulator */
    float   sum_hum;        /* %RH accumulator  */
    float   sum_press;      /* Pa accumulator    */
    uint16_t count;         /* number of samples */
} SensorAvg_t;

/* Rain pulse accumulator with daily midnight reset. */
typedef struct {
    uint32_t pulses_at_last_reset; /* snapshot of g_rain_pulses at last 24 h reset */
    uint8_t  last_reset_day;       /* day-of-month of last reset                   */
} RainTracker_t;

/* Interrupt flags and counters (defined in main.c). */
extern volatile uint8_t g_rtc_irq;
extern volatile uint8_t g_usb_irq;
extern volatile uint8_t g_rain_irq;
extern volatile uint32_t g_rain_pulses;
extern volatile uint32_t g_lora_tx_every_n;

extern SensorAvg_t  g_sensor_avg;
extern RainTracker_t g_rain_tracker;

void send_lora_payload(void);
void log_one_sample(void);
void accumulate_sensors(float t, float h_pct, float p_pa);
void OnMacProcess(void);
void OnNvmDataChange(LmHandlerNvmContextStates_t state, uint16_t size);
void OnNetworkParametersChange(CommissioningParams_t *params);
void OnMacMcpsRequest(LoRaMacStatus_t status, McpsReq_t *mcpsReq,
		TimerTime_t nextTxIn);
void OnMacMlmeRequest(LoRaMacStatus_t status, MlmeReq_t *mlmeReq,
		TimerTime_t nextTxIn);
void OnJoinRequest(LmHandlerJoinParams_t *params);
void OnTxData(LmHandlerTxParams_t *params);
void OnRxData(LmHandlerAppData_t *appData, LmHandlerRxParams_t *params);
void OnClassChange(DeviceClass_t deviceClass);
void OnBeaconStatusChange(LoRaMacHandlerBeaconParams_t *params);


void V_SENSOR_On(void);
void V_TRANS_On(void);
void V_SENSOR_Off(void);
void V_TRANS_Off(void);
bool sd_init_mount(void);
void sd_invalidate(void);
bool sd_append_line(const char *line);
bool debug_log(const char *msg);
void rtc_set_default_if_unset(void);

/* LoRaWAN ABP helpers. */
void ApplyAu915SubBand1Mask(void);

#ifdef __cplusplus
}
#endif
#endif /* INC_FUNCTIONS_H_ */
