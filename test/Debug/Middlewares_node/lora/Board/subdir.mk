################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares_node/lora/Board/adc-board.c \
../Middlewares_node/lora/Board/board.c \
../Middlewares_node/lora/Board/delay-board.c \
../Middlewares_node/lora/Board/eeprom-board.c \
../Middlewares_node/lora/Board/gpio-board.c \
../Middlewares_node/lora/Board/i2c-board.c \
../Middlewares_node/lora/Board/lpm-board.c \
../Middlewares_node/lora/Board/rtc-board.c \
../Middlewares_node/lora/Board/spi-board.c \
../Middlewares_node/lora/Board/sx1262mbxcas-board.c \
../Middlewares_node/lora/Board/sysIrqHandlers.c \
../Middlewares_node/lora/Board/uart-board.c 

OBJS += \
./Middlewares_node/lora/Board/adc-board.o \
./Middlewares_node/lora/Board/board.o \
./Middlewares_node/lora/Board/delay-board.o \
./Middlewares_node/lora/Board/eeprom-board.o \
./Middlewares_node/lora/Board/gpio-board.o \
./Middlewares_node/lora/Board/i2c-board.o \
./Middlewares_node/lora/Board/lpm-board.o \
./Middlewares_node/lora/Board/rtc-board.o \
./Middlewares_node/lora/Board/spi-board.o \
./Middlewares_node/lora/Board/sx1262mbxcas-board.o \
./Middlewares_node/lora/Board/sysIrqHandlers.o \
./Middlewares_node/lora/Board/uart-board.o 

C_DEPS += \
./Middlewares_node/lora/Board/adc-board.d \
./Middlewares_node/lora/Board/board.d \
./Middlewares_node/lora/Board/delay-board.d \
./Middlewares_node/lora/Board/eeprom-board.d \
./Middlewares_node/lora/Board/gpio-board.d \
./Middlewares_node/lora/Board/i2c-board.d \
./Middlewares_node/lora/Board/lpm-board.d \
./Middlewares_node/lora/Board/rtc-board.d \
./Middlewares_node/lora/Board/spi-board.d \
./Middlewares_node/lora/Board/sx1262mbxcas-board.d \
./Middlewares_node/lora/Board/sysIrqHandlers.d \
./Middlewares_node/lora/Board/uart-board.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares_node/lora/Board/%.o Middlewares_node/lora/Board/%.su Middlewares_node/lora/Board/%.cyclo: ../Middlewares_node/lora/Board/%.c Middlewares_node/lora/Board/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DREGION_AU915 -DSOFT_SE -DUSE_HAL_DRIVER -DSTM32L476xx -c -I../USB_DEVICE/App -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps/LmHandler" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps/LmHandler/packages" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board/EEPROM_Emul/Core" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board/EEPROM_Emul/Porting/STM32L4" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Crypto" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Inc" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Mac" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Mac/region" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Radio" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Radio/sx126x" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/System" -I../USB_DEVICE/Target -I../Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Device/ST/STM32L4xx/Include -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Include -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node" -I../FATFS/Target -I../FATFS/App -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/Third_Party/FatFs/src -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/Inc" -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares_node-2f-lora-2f-Board

clean-Middlewares_node-2f-lora-2f-Board:
	-$(RM) ./Middlewares_node/lora/Board/adc-board.cyclo ./Middlewares_node/lora/Board/adc-board.d ./Middlewares_node/lora/Board/adc-board.o ./Middlewares_node/lora/Board/adc-board.su ./Middlewares_node/lora/Board/board.cyclo ./Middlewares_node/lora/Board/board.d ./Middlewares_node/lora/Board/board.o ./Middlewares_node/lora/Board/board.su ./Middlewares_node/lora/Board/delay-board.cyclo ./Middlewares_node/lora/Board/delay-board.d ./Middlewares_node/lora/Board/delay-board.o ./Middlewares_node/lora/Board/delay-board.su ./Middlewares_node/lora/Board/eeprom-board.cyclo ./Middlewares_node/lora/Board/eeprom-board.d ./Middlewares_node/lora/Board/eeprom-board.o ./Middlewares_node/lora/Board/eeprom-board.su ./Middlewares_node/lora/Board/gpio-board.cyclo ./Middlewares_node/lora/Board/gpio-board.d ./Middlewares_node/lora/Board/gpio-board.o ./Middlewares_node/lora/Board/gpio-board.su ./Middlewares_node/lora/Board/i2c-board.cyclo ./Middlewares_node/lora/Board/i2c-board.d ./Middlewares_node/lora/Board/i2c-board.o ./Middlewares_node/lora/Board/i2c-board.su ./Middlewares_node/lora/Board/lpm-board.cyclo ./Middlewares_node/lora/Board/lpm-board.d ./Middlewares_node/lora/Board/lpm-board.o ./Middlewares_node/lora/Board/lpm-board.su ./Middlewares_node/lora/Board/rtc-board.cyclo ./Middlewares_node/lora/Board/rtc-board.d ./Middlewares_node/lora/Board/rtc-board.o ./Middlewares_node/lora/Board/rtc-board.su ./Middlewares_node/lora/Board/spi-board.cyclo ./Middlewares_node/lora/Board/spi-board.d ./Middlewares_node/lora/Board/spi-board.o ./Middlewares_node/lora/Board/spi-board.su ./Middlewares_node/lora/Board/sx1262mbxcas-board.cyclo ./Middlewares_node/lora/Board/sx1262mbxcas-board.d ./Middlewares_node/lora/Board/sx1262mbxcas-board.o ./Middlewares_node/lora/Board/sx1262mbxcas-board.su ./Middlewares_node/lora/Board/sysIrqHandlers.cyclo ./Middlewares_node/lora/Board/sysIrqHandlers.d ./Middlewares_node/lora/Board/sysIrqHandlers.o ./Middlewares_node/lora/Board/sysIrqHandlers.su ./Middlewares_node/lora/Board/uart-board.cyclo ./Middlewares_node/lora/Board/uart-board.d ./Middlewares_node/lora/Board/uart-board.o ./Middlewares_node/lora/Board/uart-board.su

.PHONY: clean-Middlewares_node-2f-lora-2f-Board

