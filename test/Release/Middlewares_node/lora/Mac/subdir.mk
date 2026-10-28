################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares_node/lora/Mac/LoRaMac.c \
../Middlewares_node/lora/Mac/LoRaMacAdr.c \
../Middlewares_node/lora/Mac/LoRaMacClassB.c \
../Middlewares_node/lora/Mac/LoRaMacCommands.c \
../Middlewares_node/lora/Mac/LoRaMacConfirmQueue.c \
../Middlewares_node/lora/Mac/LoRaMacCrypto.c \
../Middlewares_node/lora/Mac/LoRaMacParser.c \
../Middlewares_node/lora/Mac/LoRaMacSerializer.c 

OBJS += \
./Middlewares_node/lora/Mac/LoRaMac.o \
./Middlewares_node/lora/Mac/LoRaMacAdr.o \
./Middlewares_node/lora/Mac/LoRaMacClassB.o \
./Middlewares_node/lora/Mac/LoRaMacCommands.o \
./Middlewares_node/lora/Mac/LoRaMacConfirmQueue.o \
./Middlewares_node/lora/Mac/LoRaMacCrypto.o \
./Middlewares_node/lora/Mac/LoRaMacParser.o \
./Middlewares_node/lora/Mac/LoRaMacSerializer.o 

C_DEPS += \
./Middlewares_node/lora/Mac/LoRaMac.d \
./Middlewares_node/lora/Mac/LoRaMacAdr.d \
./Middlewares_node/lora/Mac/LoRaMacClassB.d \
./Middlewares_node/lora/Mac/LoRaMacCommands.d \
./Middlewares_node/lora/Mac/LoRaMacConfirmQueue.d \
./Middlewares_node/lora/Mac/LoRaMacCrypto.d \
./Middlewares_node/lora/Mac/LoRaMacParser.d \
./Middlewares_node/lora/Mac/LoRaMacSerializer.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares_node/lora/Mac/%.o Middlewares_node/lora/Mac/%.su Middlewares_node/lora/Mac/%.cyclo: ../Middlewares_node/lora/Mac/%.c Middlewares_node/lora/Mac/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DDEBUG -DREGION_AU915 -DSOFT_SE -DUSE_HAL_DRIVER -DSTM32L476xx -c -I../USB_DEVICE/App -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps/LmHandler" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps/LmHandler/packages" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board/EEPROM_Emul/Core" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board/EEPROM_Emul/Porting/STM32L4" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Crypto" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Inc" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Mac" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Mac/region" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Radio" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Radio/sx126x" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/System" -I../USB_DEVICE/Target -I../Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Device/ST/STM32L4xx/Include -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Include -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node" -I../FATFS/Target -I../FATFS/App -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/Third_Party/FatFs/src -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/Inc" -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares_node-2f-lora-2f-Mac

clean-Middlewares_node-2f-lora-2f-Mac:
	-$(RM) ./Middlewares_node/lora/Mac/LoRaMac.cyclo ./Middlewares_node/lora/Mac/LoRaMac.d ./Middlewares_node/lora/Mac/LoRaMac.o ./Middlewares_node/lora/Mac/LoRaMac.su ./Middlewares_node/lora/Mac/LoRaMacAdr.cyclo ./Middlewares_node/lora/Mac/LoRaMacAdr.d ./Middlewares_node/lora/Mac/LoRaMacAdr.o ./Middlewares_node/lora/Mac/LoRaMacAdr.su ./Middlewares_node/lora/Mac/LoRaMacClassB.cyclo ./Middlewares_node/lora/Mac/LoRaMacClassB.d ./Middlewares_node/lora/Mac/LoRaMacClassB.o ./Middlewares_node/lora/Mac/LoRaMacClassB.su ./Middlewares_node/lora/Mac/LoRaMacCommands.cyclo ./Middlewares_node/lora/Mac/LoRaMacCommands.d ./Middlewares_node/lora/Mac/LoRaMacCommands.o ./Middlewares_node/lora/Mac/LoRaMacCommands.su ./Middlewares_node/lora/Mac/LoRaMacConfirmQueue.cyclo ./Middlewares_node/lora/Mac/LoRaMacConfirmQueue.d ./Middlewares_node/lora/Mac/LoRaMacConfirmQueue.o ./Middlewares_node/lora/Mac/LoRaMacConfirmQueue.su ./Middlewares_node/lora/Mac/LoRaMacCrypto.cyclo ./Middlewares_node/lora/Mac/LoRaMacCrypto.d ./Middlewares_node/lora/Mac/LoRaMacCrypto.o ./Middlewares_node/lora/Mac/LoRaMacCrypto.su ./Middlewares_node/lora/Mac/LoRaMacParser.cyclo ./Middlewares_node/lora/Mac/LoRaMacParser.d ./Middlewares_node/lora/Mac/LoRaMacParser.o ./Middlewares_node/lora/Mac/LoRaMacParser.su ./Middlewares_node/lora/Mac/LoRaMacSerializer.cyclo ./Middlewares_node/lora/Mac/LoRaMacSerializer.d ./Middlewares_node/lora/Mac/LoRaMacSerializer.o ./Middlewares_node/lora/Mac/LoRaMacSerializer.su

.PHONY: clean-Middlewares_node-2f-lora-2f-Mac

