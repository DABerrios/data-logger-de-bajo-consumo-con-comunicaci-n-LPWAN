/**
 ******************************************************************************
 * @file    functions.c
 * @author  Daisy Berrios
 * @date    Jan 13, 2026
 * @brief   Helper functions for logging, configuration, data packaging
 *******************************************************************************
 */
#include "functions.h"
#include "main.h"
#include "ff.h"
#include "fatfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

/* Hardware and sensor drivers */
#include "rv3028.h"
#include "bme280.h"
#include "tmp117.h"
#include "diskio.h"
#include "user_diskio_spi.h"
#include "board.h"

#include "LmHandler.h"
#include "LmhPackage.h"
#include "LmhpCompliance.h"
#include "timer.h"
#include "LoRaMac.h"
#include "radio.h"
#include "sx126x.h"
#include "sx126x-board.h"
#include "Commissioning.h"
#include "board-config.h"

extern I2C_HandleTypeDef hi2c2;
extern SPI_HandleTypeDef hspi2;
extern RTC_HandleTypeDef hrtc;

static bool sd_ready = false;
static FATFS sd_fs;   /* Dedicated FATFS instance (not the CubeMX USERFatFS). */

/** Invalidates the SD card state so the next access performs a full re-init.
 *  Must be called after every V_SENSOR power cycle. */
void sd_invalidate(void) {
	sd_ready = false;
}

rv3028_t rtc_dev;

/* ---------- Sensor averaging --------------------------------------------- */
SensorAvg_t  g_sensor_avg   = { 0 };
RainTracker_t g_rain_tracker = { 0 };

void accumulate_sensors(float t, float h_pct, float p_pa)
{
	g_sensor_avg.sum_temp  += t;
	g_sensor_avg.sum_hum   += h_pct;
	g_sensor_avg.sum_press += p_pa;
	g_sensor_avg.count++;
}

/* ---------- SD-card averaging: read last N lines of DATA.csv ------------ */

/**
 * Reads the tail of DATA.csv and computes the average of the last n samples.
 *
 * CSV format (written by log_one_sample):
 *   rain,temp,hum,press,HH:MM:SS,DD/MM/YYYY
 *
 * Fields: [0] rain (uint32), [1] temp (deg C), [2] humidity (%RH),
 *         [3] pressure (hPa).
 *
 * @param  n      Number of most-recent lines to average.
 * @param  avg_t  Output: mean temperature (deg C).
 * @param  avg_h  Output: mean humidity (%RH).
 * @param  avg_p  Output: mean pressure (hPa).
 * @return Number of lines successfully parsed (0 on failure).
 */
#define SD_TAIL_BUF_SIZE  640   /* approximately 10 lines at 60 bytes each */

static uint16_t sd_read_last_n_averages(uint16_t n,
                                        float *avg_t, float *avg_h, float *avg_p)
{
	FIL fil;
	FRESULT fr;
	UINT br;
	char buf[SD_TAIL_BUF_SIZE];
	FSIZE_t fsize;
	FSIZE_t read_start;
	uint16_t parsed = 0;
	float sum_t = 0.0f, sum_h = 0.0f, sum_p = 0.0f;

	*avg_t = 0.0f;
	*avg_h = 0.0f;
	*avg_p = 0.0f;

	if (!sd_init_mount()) {
		return 0;
	}

	fr = f_open(&fil, "0:/DATA.csv", FA_READ);
	if (fr != FR_OK) {
		f_mount(NULL, "0:/", 0);
		sd_ready = false;
		return 0;
	}

	fsize = f_size(&fil);
	if (fsize == 0) {
		f_close(&fil);
		f_mount(NULL, "0:/", 0);
		sd_ready = false;
		return 0;
	}

	/* Read tail chunk of the file */
	if (fsize <= SD_TAIL_BUF_SIZE) {
		read_start = 0;
	} else {
		read_start = fsize - SD_TAIL_BUF_SIZE;
	}

	fr = f_lseek(&fil, read_start);
	if (fr != FR_OK) { f_close(&fil); f_mount(NULL, "0:/", 0); sd_ready = false; return 0; }

	fr = f_read(&fil, buf, SD_TAIL_BUF_SIZE - 1, &br);
	f_close(&fil);

	/* Unmount after every operation to ensure clean FATFS state. */
	f_mount(NULL, "0:/", 0);
	sd_ready = false;

	if (fr != FR_OK || br == 0)
		return 0;

	buf[br] = '\0';

	/* Collect pointers to the start of each complete line.
	 * If reading began mid-file, the first partial line is skipped. */
	char *lines[32];
	uint16_t line_count = 0;
	char *p = buf;

	/* If we read from mid-file, skip the first partial line */
	if (read_start > 0) {
		p = strchr(buf, '\n');
		if (p == NULL) return 0;
		p++;  /* skip past the \n */
	}

	/* Collect line start pointers */
	while (*p != '\0' && line_count < 32) {
		/* Skip the CSV header if encountered */
		if (strncmp(p, "rain,", 5) == 0) {
			char *nl = strchr(p, '\n');
			if (nl) p = nl + 1;
			else break;
			continue;
		}

		char *nl = strchr(p, '\n');
		if (nl == NULL) break;  /* incomplete line at end of buffer */

		lines[line_count++] = p;
		p = nl + 1;
	}

	if (line_count == 0)
		return 0;

	/* Take the last N lines (or fewer if not enough) */
	uint16_t start_idx = (line_count > n) ? (line_count - n) : 0;

	for (uint16_t i = start_idx; i < line_count; i++) {
		/* Parse CSV fields using strtof (newlib-nano lacks scanf %f support). */
		char *cur = lines[i];
		char *comma;

		/* Skip field 0 (rain — live counter is used instead). */
		comma = strchr(cur, ',');
		if (comma == NULL) continue;
		cur = comma + 1;

		/* Field 1: temperature (deg C). */
		comma = strchr(cur, ',');
		if (comma == NULL) continue;
		*comma = '\0';
		float t_val = strtof(cur, NULL);
		*comma = ',';
		cur = comma + 1;

		/* Field 2: humidity (%RH). */
		comma = strchr(cur, ',');
		if (comma == NULL) continue;
		*comma = '\0';
		float h_val = strtof(cur, NULL);
		*comma = ',';
		cur = comma + 1;

		/* Field 3: pressure (hPa). */
		comma = strchr(cur, ',');
		if (comma == NULL) continue;
		*comma = '\0';
		float p_val = strtof(cur, NULL);
		*comma = ',';

		/* Validate ranges before including in the average. */
		if (t_val >= -40.0f && t_val <= 85.0f &&
		    h_val >= 0.0f   && h_val <= 100.0f &&
		    p_val >= 300.0f && p_val <= 1150.0f) {
			sum_t += t_val;
			sum_h += h_val;
			sum_p += p_val;
			parsed++;
		}
	}

	if (parsed > 0) {
		*avg_t = sum_t / parsed;
		*avg_h = sum_h / parsed;
		*avg_p = sum_p / parsed;
	}

	return parsed;
}

/** Builds and transmits a 7-byte big-endian LoRaWAN uplink payload.
 *  Averages the last N sensor readings from the SD card and includes
 *  the cumulative daily rain pulse count. */
void send_lora_payload(void)
{
	uint8_t buf[LORA_PAYLOAD_SIZE];
	uint32_t rain_raw;
	float avg_t, avg_h, avg_p;

	/* Read averaged sensor data from the SD card. */
	uint16_t n_samples = sd_read_last_n_averages(
			g_lora_tx_every_n > 0 ? g_lora_tx_every_n : 1,
			&avg_t, &avg_h, &avg_p);

	if (n_samples == 0) {
		return;
	}

	/* Compute cumulative rain pulses since the last daily reset. */
	__disable_irq();
	rain_raw = g_rain_pulses;
	__enable_irq();

	{
		rv3028_datetime_t now;
		rv3028_get_datetime(&rtc_dev, &now);

		if (g_rain_tracker.last_reset_day == 0) {
			g_rain_tracker.pulses_at_last_reset = rain_raw;
			g_rain_tracker.last_reset_day = now.date;
		}

		if (now.date != g_rain_tracker.last_reset_day) {
			g_rain_tracker.pulses_at_last_reset = rain_raw;
			g_rain_tracker.last_reset_day = now.date;
		}
	}

	uint16_t rain_today = (uint16_t)(rain_raw - g_rain_tracker.pulses_at_last_reset);

	/* Encode sensor values into the 7-byte payload.
	 * Temperature: deg C x 100 (int16), Humidity: %RH (uint8),
	 * Pressure: hPa x 10 (uint16). */
	int16_t  temp_enc = (int16_t)(avg_t * 100.0f);
	uint8_t  hum_enc  = (uint8_t)(avg_h + 0.5f);
	uint16_t pres_enc = (uint16_t)(avg_p * 10.0f + 0.5f);

	buf[0] = (uint8_t)(rain_today >> 8);
	buf[1] = (uint8_t)(rain_today);
	buf[2] = (uint8_t)((uint16_t)temp_enc >> 8);
	buf[3] = (uint8_t)((uint16_t)temp_enc);
	buf[4] = hum_enc;
	buf[5] = (uint8_t)(pres_enc >> 8);
	buf[6] = (uint8_t)(pres_enc);

	LmHandlerAppData_t appData = { 0 };
	appData.Buffer     = buf;
	appData.BufferSize = LORA_PAYLOAD_SIZE;
	appData.Port       = LORAWAN_APP_PORT;

	if (LmHandlerIsBusy() == false) {
		LmHandlerSend(&appData, LORAMAC_HANDLER_UNCONFIRMED_MSG);
	}
}

/* ---------- LoRaWAN handler callbacks ------------------------------------ */

/** Called when an ABP join (personalisation) completes. Retries on failure. */
void OnJoinRequest(LmHandlerJoinParams_t *params) {
	if (params->Status == LORAMAC_HANDLER_ERROR) {
		LmHandlerJoin();
	}
}

/** Transmission-complete callback (unused in fire-and-forget mode). */
void OnTxData(LmHandlerTxParams_t *params) {
	(void)params;
}

/** Downlink reception callback (unused — ABP TX-only operation). */
void OnRxData(LmHandlerAppData_t *appData, LmHandlerRxParams_t *params) {
	(void)appData;
	(void)params;
}

/** MAC processing notification (triggers LmHandlerProcess in the main loop). */
void OnMacProcess(void) {
}

void OnNvmDataChange(LmHandlerNvmContextStates_t state, uint16_t size) {
	(void)state;
	(void)size;
}

void OnNetworkParametersChange(CommissioningParams_t *params) {
	(void)params;
}

void OnMacMcpsRequest(LoRaMacStatus_t status, McpsReq_t *mcpsReq,
		TimerTime_t nextTxIn) {
	(void)status;
	(void)mcpsReq;
	(void)nextTxIn;
}

void OnMacMlmeRequest(LoRaMacStatus_t status, MlmeReq_t *mlmeReq,
		TimerTime_t nextTxIn) {
	(void)status;
	(void)mlmeReq;
	(void)nextTxIn;
}

void OnClassChange(DeviceClass_t deviceClass) {
	(void)deviceClass;
}

void OnBeaconStatusChange(LoRaMacHandlerBeaconParams_t *params) {
	(void)params;
}

/** Configures the AU915 channel mask to enable only 500 kHz channel 64
 *  (915.9 MHz, SF8). All 125 kHz channels are disabled. */
void ApplyAu915SubBand1Mask(void) {
	MibRequestConfirm_t mibReq;
	static uint16_t channelMask[6] = { 0 };

	/* Disable ALL 125kHz channels */
	channelMask[0] = 0x0000;
	channelMask[1] = 0x0000;
	channelMask[2] = 0x0000;
	channelMask[3] = 0x0000;
	/* Enable ONLY 500kHz channel 64 (bit 0 of mask[4]) */
	channelMask[4] = 0x0001;
	channelMask[5] = 0x0000;

	mibReq.Type = MIB_CHANNELS_MASK;
	mibReq.Param.ChannelsMask = channelMask;
	LoRaMacMibSetRequestConfirm(&mibReq);

	mibReq.Type = MIB_CHANNELS_DEFAULT_MASK;
	mibReq.Param.ChannelsDefaultMask = channelMask;
	LoRaMacMibSetRequestConfirm(&mibReq);
}

/* ---------- Power rail control ------------------------------------------- */

/** Enables the V_SENSOR load switch and waits for the rail to stabilise. */
void V_SENSOR_On(void) {
	HAL_GPIO_WritePin(EN_v_sensor_GPIO_Port, EN_v_sensor_Pin, GPIO_PIN_SET);
	HAL_Delay(3);
}

/** Disables the V_SENSOR load switch. */
void V_SENSOR_Off(void) {
	HAL_GPIO_WritePin(EN_v_sensor_GPIO_Port, EN_v_sensor_Pin, GPIO_PIN_RESET);
}

/** Enables the V_TRANS load switch and waits for the rail to stabilise. */
void V_TRANS_On(void) {
	HAL_GPIO_WritePin(EN_v_trans_GPIO_Port, EN_v_trans_Pin, GPIO_PIN_SET);
	HAL_Delay(3);
}

/** Disables the V_TRANS load switch. */
void V_TRANS_Off(void) {
	HAL_GPIO_WritePin(EN_v_trans_GPIO_Port, EN_v_trans_Pin, GPIO_PIN_RESET);
}

/* ---------- SD card helpers ---------------------------------------------- */

/** Mounts the SD card, clearing all three layers of initialisation caching
 *  (CubeMX ff_gen_drv, user_diskio_spi Stat, and the local sd_ready flag).
 *  This ensures a full hardware re-init after every V_SENSOR power cycle. */
bool sd_init_mount(void) {
	FRESULT fr;

	{
		extern Disk_drvTypeDef disk;
		disk.is_initialized[0] = 0;
	}
	USER_SPI_force_reinit();

	memset(&sd_fs, 0, sizeof(sd_fs));
	fr = f_mount(&sd_fs, "0:/", 1);
	if (fr != FR_OK) {
		HAL_RTCEx_BKUPWrite(&hrtc, RTC_BKP_DR3, 0xFA00 | (fr & 0xFF));
		return false;
	}

	sd_ready = true;
	return true;
}

/** Sets the external RTC to a default date if it reports year 2000 or earlier. */
void rtc_set_default_if_unset(void) {
	rv3028_datetime_t now;

	if (!rv3028_init(&rtc_dev, &hi2c2, RV3028_I2C_ADDR_DEFAULT)) {
		return;
	}
	if (!rv3028_get_datetime(&rtc_dev, &now))
		return;

	if (now.year == 2000 || now.year < 2024) {
		rv3028_datetime_t def = { .year = 2026, .month = 1, .date = 20, .hours =
				12, .minutes = 0, .seconds = 0, .weekday = 2 };

		(void) rv3028_set_datetime(&rtc_dev, &def);
	}
}

static const char *CSV_HEADER = "rain,temp,hum,press,date,time\r\n";

/** Appends a line to DATA.csv, creating the file with a header if needed.
 *  Unmounts the SD card after every write for clean state on power cycle. */
bool sd_append_line(const char *line) {
	FIL fil;
	UINT bw;
	FRESULT fr;

	if (!sd_init_mount())
		return false;

	fr = f_open(&fil, "0:/DATA.csv", FA_OPEN_ALWAYS | FA_WRITE);
	if (fr != FR_OK) {
		f_mount(NULL, "0:/", 0);
		sd_ready = false;
		return false;
	}

	if (f_size(&fil) == 0) {
		fr = f_write(&fil, CSV_HEADER, (UINT) strlen(CSV_HEADER), &bw);
	}

	fr = f_lseek(&fil, f_size(&fil));
	fr = f_write(&fil, line, (UINT) strlen(line), &bw);
	fr = f_sync(&fil);
	f_close(&fil);

	f_mount(NULL, "0:/", 0);
	sd_ready = false;

	return (fr == FR_OK);
}

/** Appends a debug message to DEBUG.txt on the SD card. */
bool debug_log(const char *msg) {
	FIL fil;
	UINT bw;
	FRESULT fr;

	if (!sd_init_mount())
		return false;

	fr = f_open(&fil, "0:/DEBUG.txt", FA_OPEN_ALWAYS | FA_WRITE);
	if (fr != FR_OK) {
		f_mount(NULL, "0:/", 0);
		sd_ready = false;
		return false;
	}

	fr = f_lseek(&fil, f_size(&fil));
	fr = f_write(&fil, msg, (UINT) strlen(msg), &bw);
	fr = f_sync(&fil);
	f_close(&fil);

	f_mount(NULL, "0:/", 0);
	sd_ready = false;

	return (fr == FR_OK);
}

/** Reads all sensors and appends one timestamped CSV line to DATA.csv.
 *  send_lora_payload() later reads the last N lines for averaging. */
void log_one_sample(void) {
	rv3028_datetime_t now;
	bme280_data_t bme;
	float t_c = 0.0f;
	uint32_t rain;

	__disable_irq();
	rain = g_rain_pulses;
	__enable_irq();

	rv3028_get_datetime(&rtc_dev, &now);
	HAL_Delay(20);
	bme280_read_forced(&hi2c2, &bme);
	tmp117_oneshot_read_temperature_c(&hi2c2, &t_c, 200);

	char log_buffer[128];
	snprintf(log_buffer, sizeof(log_buffer),
			"%lu,%.2f,%.2f,%.2f,%02d:%02d:%02d,%02d/%02d/%04d\r\n", rain, t_c,
			bme.humidity_rh, bme.pressure_pa / 100.0f, now.hours, now.minutes,
			now.seconds, now.date, now.month, now.year);

	sd_append_line(log_buffer);
}

                                                                                                                                                                                                                 