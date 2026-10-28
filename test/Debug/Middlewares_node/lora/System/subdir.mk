################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares_node/lora/System/adc.c \
../Middlewares_node/lora/System/delay.c \
../Middlewares_node/lora/System/fifo.c \
../Middlewares_node/lora/System/gpio.c \
../Middlewares_node/lora/System/i2c.c \
../Middlewares_node/lora/System/nvmm.c \
../Middlewares_node/lora/System/systime.c \
../Middlewares_node/lora/System/timer.c \
../Middlewares_node/lora/System/uart.c \
../Middlewares_node/lora/System/utilities.c 

OBJS += \
./Middlewares_node/lora/System/adc.o \
./Middlewares_node/lora/System/delay.o \
./Middlewares_node/lora/System/fifo.o \
./Middlewares_node/lora/System/gpio.o \
./Middlewares_node/lora/System/i2c.o \
./Middlewares_node/lora/System/nvmm.o \
./Middlewares_node/lora/System/systime.o \
./Middlewares_node/lora/System/timer.o \
./Middlewares_node/lora/System/uart.o \
./Middlewares_node/lora/System/utilities.o 

C_DEPS += \
./Middlewares_node/lora/System/adc.d \
./Middlewares_node/lora/System/delay.d \
./Middlewares_node/lora/System/fifo.d \
./Middlewares_node/lora/System/gpio.d \
./Middlewares_node/lora/System/i2c.d \
./Middlewares_node/lora/System/nvmm.d \
./Middlewares_node/lora/System/systime.d \
./Middlewares_node/lora/System/timer.d \
./Middlewares_node/lora/System/uart.d \
./Middlewares_node/lora/System/utilities.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares_node/lora/System/%.o Middlewares_node/lora/System/%.su Middlewares_node/lora/System/%.cyclo: ../Middlewares_node/lora/System/%.c Middlewares_node/lora/System/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DREGION_AU915 -DSOFT_SE -DUSE_HAL_DRIVER -DSTM32L476xx -c -I../USB_DEVICE/App -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps/LmHandler" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps/LmHandler/packages" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board/EEPROM_Emul/Core" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board/EEPROM_Emul/Porting/STM32L4" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Crypto" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Inc" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Mac" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Mac/region" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Radio" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Radio/sx126x" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/System" -I../USB_DEVICE/Target -I../Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Device/ST/STM32L4xx/Include -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Include -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node" -I../FATFS/Target -I../FATFS/App -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/Third_Party/FatFs/src -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/Inc" -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares_node-2f-lora-2f-System

clean-Middlewares_node-2f-lora-2f-System:
	-$(RM) ./Middlewares_node/lora/System/adc.cyclo ./Middlewares_node/lora/System/adc.d ./Middlewares_node/lora/System/adc.o ./Middlewares_node/lora/System/adc.su ./Middlewares_node/lora/System/delay.cyclo ./Middlewares_node/lora/System/delay.d ./Middlewares_node/lora/System/delay.o ./Middlewares_node/lora/System/delay.su ./Middlewares_node/lora/System/fifo.cyclo ./Middlewares_node/lora/System/fifo.d ./Middlewares_node/lora/System/fifo.o ./Middlewares_node/lora/System/fifo.su ./Middlewares_node/lora/System/gpio.cyclo ./Middlewares_node/lora/System/gpio.d ./Middlewares_node/lora/System/gpio.o ./Middlewares_node/lora/System/gpio.su ./Middlewares_node/lora/System/i2c.cyclo ./Middlewares_node/lora/System/i2c.d ./Middlewares_node/lora/System/i2c.o ./Middlewares_node/lora/System/i2c.su ./Middlewares_node/lora/System/nvmm.cyclo ./Middlewares_node/lora/System/nvmm.d ./Middlewares_node/lora/System/nvmm.o ./Middlewares_node/lora/System/nvmm.su ./Middlewares_node/lora/System/systime.cyclo ./Middlewares_node/lora/System/systime.d ./Middlewares_node/lora/System/systime.o ./Middlewares_node/lora/System/systime.su ./Middlewares_node/lora/System/timer.cyclo ./Middlewares_node/lora/System/timer.d ./Middlewares_node/lora/System/timer.o ./Middlewares_node/lora/System/timer.su ./Middlewares_node/lora/System/uart.cyclo ./Middlewares_node/lora/System/uart.d ./Middlewares_node/lora/System/uart.o ./Middlewares_node/lora/System/uart.su ./Middlewares_node/lora/System/utilities.cyclo ./Middlewares_node/lora/System/utilities.d ./Middlewares_node/lora/System/utilities.o ./Middlewares_node/lora/System/utilities.su

.PHONY: clean-Middlewares_node-2f-lora-2f-System

