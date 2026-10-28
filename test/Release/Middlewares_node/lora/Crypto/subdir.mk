################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares_node/lora/Crypto/aes.c \
../Middlewares_node/lora/Crypto/cmac.c \
../Middlewares_node/lora/Crypto/soft-se-hal.c \
../Middlewares_node/lora/Crypto/soft-se.c 

OBJS += \
./Middlewares_node/lora/Crypto/aes.o \
./Middlewares_node/lora/Crypto/cmac.o \
./Middlewares_node/lora/Crypto/soft-se-hal.o \
./Middlewares_node/lora/Crypto/soft-se.o 

C_DEPS += \
./Middlewares_node/lora/Crypto/aes.d \
./Middlewares_node/lora/Crypto/cmac.d \
./Middlewares_node/lora/Crypto/soft-se-hal.d \
./Middlewares_node/lora/Crypto/soft-se.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares_node/lora/Crypto/%.o Middlewares_node/lora/Crypto/%.su Middlewares_node/lora/Crypto/%.cyclo: ../Middlewares_node/lora/Crypto/%.c Middlewares_node/lora/Crypto/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DDEBUG -DREGION_AU915 -DSOFT_SE -DUSE_HAL_DRIVER -DSTM32L476xx -c -I../USB_DEVICE/App -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps/LmHandler" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps/LmHandler/packages" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board/EEPROM_Emul/Core" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board/EEPROM_Emul/Porting/STM32L4" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Crypto" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Inc" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Mac" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Mac/region" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Radio" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Radio/sx126x" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/System" -I../USB_DEVICE/Target -I../Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Device/ST/STM32L4xx/Include -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Include -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node" -I../FATFS/Target -I../FATFS/App -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/Third_Party/FatFs/src -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/Inc" -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares_node-2f-lora-2f-Crypto

clean-Middlewares_node-2f-lora-2f-Crypto:
	-$(RM) ./Middlewares_node/lora/Crypto/aes.cyclo ./Middlewares_node/lora/Crypto/aes.d ./Middlewares_node/lora/Crypto/aes.o ./Middlewares_node/lora/Crypto/aes.su ./Middlewares_node/lora/Crypto/cmac.cyclo ./Middlewares_node/lora/Crypto/cmac.d ./Middlewares_node/lora/Crypto/cmac.o ./Middlewares_node/lora/Crypto/cmac.su ./Middlewares_node/lora/Crypto/soft-se-hal.cyclo ./Middlewares_node/lora/Crypto/soft-se-hal.d ./Middlewares_node/lora/Crypto/soft-se-hal.o ./Middlewares_node/lora/Crypto/soft-se-hal.su ./Middlewares_node/lora/Crypto/soft-se.cyclo ./Middlewares_node/lora/Crypto/soft-se.d ./Middlewares_node/lora/Crypto/soft-se.o ./Middlewares_node/lora/Crypto/soft-se.su

.PHONY: clean-Middlewares_node-2f-lora-2f-Crypto

