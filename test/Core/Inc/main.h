/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32l4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define rain_int_Pin GPIO_PIN_13
#define rain_int_GPIO_Port GPIOC
#define rain_int_EXTI_IRQn EXTI15_10_IRQn
#define SD_MISO_Pin GPIO_PIN_2
#define SD_MISO_GPIO_Port GPIOC
#define EN_v_trans_Pin GPIO_PIN_1
#define EN_v_trans_GPIO_Port GPIOA
#define EN_v_sensor_Pin GPIO_PIN_2
#define EN_v_sensor_GPIO_Port GPIOA
#define LORA_CS_Pin GPIO_PIN_4
#define LORA_CS_GPIO_Port GPIOC
#define LORA_rst_Pin GPIO_PIN_5
#define LORA_rst_GPIO_Port GPIOC
#define LORA_DIO_Pin GPIO_PIN_0
#define LORA_DIO_GPIO_Port GPIOB
#define LORA_DIO_EXTI_IRQn EXTI0_IRQn
#define LORA_BUSY_Pin GPIO_PIN_1
#define LORA_BUSY_GPIO_Port GPIOB
#define Lora_ANT_SW_Pin GPIO_PIN_2
#define Lora_ANT_SW_GPIO_Port GPIOB
#define SD_SCK_Pin GPIO_PIN_10
#define SD_SCK_GPIO_Port GPIOB
#define SD_CS_Pin GPIO_PIN_11
#define SD_CS_GPIO_Port GPIOB
#define SD_MOSI_Pin GPIO_PIN_15
#define SD_MOSI_GPIO_Port GPIOB
#define RTC_int_Pin GPIO_PIN_8
#define RTC_int_GPIO_Port GPIOA
#define RTC_int_EXTI_IRQn EXTI9_5_IRQn
#define BQ25570_VBATok_Pin GPIO_PIN_15
#define BQ25570_VBATok_GPIO_Port GPIOA
#define BQ25570_Vout_EN_Pin GPIO_PIN_10
#define BQ25570_Vout_EN_GPIO_Port GPIOC
#define usb_wakeup_Pin GPIO_PIN_6
#define usb_wakeup_GPIO_Port GPIOB
#define usb_wakeup_EXTI_IRQn EXTI9_5_IRQn
#define usb_detect_Pin GPIO_PIN_4
#define usb_detect_GPIO_Port GPIOB
#define usb_detect_EXTI_IRQn EXTI4_IRQn

/* USER CODE BEGIN Private defines */
extern SPI_HandleTypeDef hspi2;
#define SD_SPI_HANDLE hspi2
/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
