################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares_node/Src/bme280.c \
../Middlewares_node/Src/microshell.c \
../Middlewares_node/Src/rv3028.c \
../Middlewares_node/Src/tmp117.c \
../Middlewares_node/Src/user_diskio_spi.c 

OBJS += \
./Middlewares_node/Src/bme280.o \
./Middlewares_node/Src/microshell.o \
./Middlewares_node/Src/rv3028.o \
./Middlewares_node/Src/tmp117.o \
./Middlewares_node/Src/user_diskio_spi.o 

C_DEPS += \
./Middlewares_node/Src/bme280.d \
./Middlewares_node/Src/microshell.d \
./Middlewares_node/Src/rv3028.d \
./Middlewares_node/Src/tmp117.d \
./Middlewares_node/Src/user_diskio_spi.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares_node/Src/%.o Middlewares_node/Src/%.su Middlewares_node/Src/%.cyclo: ../Middlewares_node/Src/%.c Middlewares_node/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DDEBUG -DREGION_AU915 -DSOFT_SE -DUSE_HAL_DRIVER -DSTM32L476xx -c -I../USB_DEVICE/App -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps/LmHandler" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps/LmHandler/packages" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board/EEPROM_Emul/Core" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board/EEPROM_Emul/Porting/STM32L4" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Crypto" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Inc" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Mac" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Mac/region" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Radio" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Radio/sx126x" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/System" -I../USB_DEVICE/Target -I../Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Device/ST/STM32L4xx/Include -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Include -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node" -I../FATFS/Target -I../FATFS/App -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/Third_Party/FatFs/src -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/Inc" -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares_node-2f-Src

clean-Middlewares_node-2f-Src:
	-$(RM) ./Middlewares_node/Src/bme280.cyclo ./Middlewares_node/Src/bme280.d ./Middlewares_node/Src/bme280.o ./Middlewares_node/Src/bme280.su ./Middlewares_node/Src/microshell.cyclo ./Middlewares_node/Src/microshell.d ./Middlewares_node/Src/microshell.o ./Middlewares_node/Src/microshell.su ./Middlewares_node/Src/rv3028.cyclo ./Middlewares_node/Src/rv3028.d ./Middlewares_node/Src/rv3028.o ./Middlewares_node/Src/rv3028.su ./Middlewares_node/Src/tmp117.cyclo ./Middlewares_node/Src/tmp117.d ./Middlewares_node/Src/tmp117.o ./Middlewares_node/Src/tmp117.su ./Middlewares_node/Src/user_diskio_spi.cyclo ./Middlewares_node/Src/user_diskio_spi.d ./Middlewares_node/Src/user_diskio_spi.o ./Middlewares_node/Src/user_diskio_spi.su

.PHONY: clean-Middlewares_node-2f-Src

