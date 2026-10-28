################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/functions.c \
../Core/Src/main.c \
../Core/Src/stm32l4xx_hal_msp.c \
../Core/Src/stm32l4xx_it.c \
../Core/Src/syscalls.c \
../Core/Src/sysmem.c \
../Core/Src/system_stm32l4xx.c 

OBJS += \
./Core/Src/functions.o \
./Core/Src/main.o \
./Core/Src/stm32l4xx_hal_msp.o \
./Core/Src/stm32l4xx_it.o \
./Core/Src/syscalls.o \
./Core/Src/sysmem.o \
./Core/Src/system_stm32l4xx.o 

C_DEPS += \
./Core/Src/functions.d \
./Core/Src/main.d \
./Core/Src/stm32l4xx_hal_msp.d \
./Core/Src/stm32l4xx_it.d \
./Core/Src/syscalls.d \
./Core/Src/sysmem.d \
./Core/Src/system_stm32l4xx.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/%.o Core/Src/%.su Core/Src/%.cyclo: ../Core/Src/%.c Core/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DDEBUG -DREGION_AU915 -DSOFT_SE -DUSE_HAL_DRIVER -DSTM32L476xx -c -I../USB_DEVICE/App -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps/LmHandler" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/apps/LmHandler/packages" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board/EEPROM_Emul/Core" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Board/EEPROM_Emul/Porting/STM32L4" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Crypto" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Inc" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Mac" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Mac/region" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Radio" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/Radio/sx126x" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/lora/System" -I../USB_DEVICE/Target -I../Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Device/ST/STM32L4xx/Include -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Include -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares" -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node" -I../FATFS/Target -I../FATFS/App -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/Third_Party/FatFs/src -I"C:/Users/daisy/STM32CubeIDE/working_Ver_logging_and _sleep/test/Middlewares_node/Inc" -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src

clean-Core-2f-Src:
	-$(RM) ./Core/Src/functions.cyclo ./Core/Src/functions.d ./Core/Src/functions.o ./Core/Src/functions.su ./Core/Src/main.cyclo ./Core/Src/main.d ./Core/Src/main.o ./Core/Src/main.su ./Core/Src/stm32l4xx_hal_msp.cyclo ./Core/Src/stm32l4xx_hal_msp.d ./Core/Src/stm32l4xx_hal_msp.o ./Core/Src/stm32l4xx_hal_msp.su ./Core/Src/stm32l4xx_it.cyclo ./Core/Src/stm32l4xx_it.d ./Core/Src/stm32l4xx_it.o ./Core/Src/stm32l4xx_it.su ./Core/Src/syscalls.cyclo ./Core/Src/syscalls.d ./Core/Src/syscalls.o ./Core/Src/syscalls.su ./Core/Src/sysmem.cyclo ./Core/Src/sysmem.d ./Core/Src/sysmem.o ./Core/Src/sysmem.su ./Core/Src/system_stm32l4xx.cyclo ./Core/Src/system_stm32l4xx.d ./Core/Src/system_stm32l4xx.o ./Core/Src/system_stm32l4xx.su

.PHONY: clean-Core-2f-Src

