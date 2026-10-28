################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares_node/lora/apps/LmHandler/packages/FragDecoder.c \
../Middlewares_node/lora/apps/LmHandler/packages/LmhpClockSync.c \
../Middlewares_node/lora/apps/LmHandler/packages/LmhpCompliance.c \
../Middlewares_node/lora/apps/LmHandler/packages/LmhpFragmentation.c \
../Middlewares_node/lora/apps/LmHandler/packages/LmhpRemoteMcastSetup.c 

OBJS += \
./Middlewares_node/lora/apps/LmHandler/packages/FragDecoder.o \
./Middlewares_node/lora/apps/LmHandler/packages/LmhpClockSync.o \
./Middlewares_node/lora/apps/LmHandler/packages/LmhpCompliance.o \
./Middlewares_node/lora/apps/LmHandler/packages/LmhpFragmentation.o \
./Middlewares_node/lora/apps/LmHandler/packages/LmhpRemoteMcastSetup.o 

C_DEPS += \
./Middlewares_node/lora/apps/LmHandler/packages/FragDecoder.d \
./Middlewares_node/lora/apps/LmHandler/packages/LmhpClockSync.d \
./Middlewares_node/lora/apps/LmHandler/packages/LmhpCompliance.d \
./Middlewares_node/lora/apps/LmHandler/packages/LmhpFragmentation.d \
./Middlewares_node/lora/apps/LmHandler/packages/LmhpRemoteMcastSetup.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares_node/lora/apps/LmHandler/packages/%.o Middlewares_node/lora/apps/LmHandler/packages/%.su Middlewares_node/lora/apps/LmHandler/packages/%.cyclo: ../Middlewares_node/lora/apps/LmHandler/packages/%.c Middlewares_node/lora/apps/LmHandler/packages/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DDEBUG -DREGION_AU915 -DSOFT_SE -DUSE_HAL_DRIVER -DSTM32L476xx -c -I../USB_DEVICE/App -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps/LmHandler" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps/LmHandler/packages" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board/EEPROM_Emul/Core" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board/EEPROM_Emul/Porting/STM32L4" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Crypto" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Inc" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Mac" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Mac/region" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Radio" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Radio/sx126x" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/System" -I../USB_DEVICE/Target -I../Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Device/ST/STM32L4xx/Include -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Include -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node" -I../FATFS/Target -I../FATFS/App -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/Third_Party/FatFs/src -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/Inc" -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares_node-2f-lora-2f-apps-2f-LmHandler-2f-packages

clean-Middlewares_node-2f-lora-2f-apps-2f-LmHandler-2f-packages:
	-$(RM) ./Middlewares_node/lora/apps/LmHandler/packages/FragDecoder.cyclo ./Middlewares_node/lora/apps/LmHandler/packages/FragDecoder.d ./Middlewares_node/lora/apps/LmHandler/packages/FragDecoder.o ./Middlewares_node/lora/apps/LmHandler/packages/FragDecoder.su ./Middlewares_node/lora/apps/LmHandler/packages/LmhpClockSync.cyclo ./Middlewares_node/lora/apps/LmHandler/packages/LmhpClockSync.d ./Middlewares_node/lora/apps/LmHandler/packages/LmhpClockSync.o ./Middlewares_node/lora/apps/LmHandler/packages/LmhpClockSync.su ./Middlewares_node/lora/apps/LmHandler/packages/LmhpCompliance.cyclo ./Middlewares_node/lora/apps/LmHandler/packages/LmhpCompliance.d ./Middlewares_node/lora/apps/LmHandler/packages/LmhpCompliance.o ./Middlewares_node/lora/apps/LmHandler/packages/LmhpCompliance.su ./Middlewares_node/lora/apps/LmHandler/packages/LmhpFragmentation.cyclo ./Middlewares_node/lora/apps/LmHandler/packages/LmhpFragmentation.d ./Middlewares_node/lora/apps/LmHandler/packages/LmhpFragmentation.o ./Middlewares_node/lora/apps/LmHandler/packages/LmhpFragmentation.su ./Middlewares_node/lora/apps/LmHandler/packages/LmhpRemoteMcastSetup.cyclo ./Middlewares_node/lora/apps/LmHandler/packages/LmhpRemoteMcastSetup.d ./Middlewares_node/lora/apps/LmHandler/packages/LmhpRemoteMcastSetup.o ./Middlewares_node/lora/apps/LmHandler/packages/LmhpRemoteMcastSetup.su

.PHONY: clean-Middlewares_node-2f-lora-2f-apps-2f-LmHandler-2f-packages

