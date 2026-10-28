################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/USB_Device_Library/usbd_cdc.c \
../Middlewares/USB_Device_Library/usbd_core.c \
../Middlewares/USB_Device_Library/usbd_ctlreq.c \
../Middlewares/USB_Device_Library/usbd_ioreq.c 

OBJS += \
./Middlewares/USB_Device_Library/usbd_cdc.o \
./Middlewares/USB_Device_Library/usbd_core.o \
./Middlewares/USB_Device_Library/usbd_ctlreq.o \
./Middlewares/USB_Device_Library/usbd_ioreq.o 

C_DEPS += \
./Middlewares/USB_Device_Library/usbd_cdc.d \
./Middlewares/USB_Device_Library/usbd_core.d \
./Middlewares/USB_Device_Library/usbd_ctlreq.d \
./Middlewares/USB_Device_Library/usbd_ioreq.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/USB_Device_Library/%.o Middlewares/USB_Device_Library/%.su Middlewares/USB_Device_Library/%.cyclo: ../Middlewares/USB_Device_Library/%.c Middlewares/USB_Device_Library/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DREGION_AU915 -DSOFT_SE -DUSE_HAL_DRIVER -DSTM32L476xx -c -I../USB_DEVICE/App -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps/LmHandler" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps/LmHandler/packages" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board/EEPROM_Emul/Core" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board/EEPROM_Emul/Porting/STM32L4" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Crypto" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Inc" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Mac" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Mac/region" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Radio" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Radio/sx126x" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/System" -I../USB_DEVICE/Target -I../Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Device/ST/STM32L4xx/Include -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Include -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node" -I../FATFS/Target -I../FATFS/App -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/Third_Party/FatFs/src -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/Inc" -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-USB_Device_Library

clean-Middlewares-2f-USB_Device_Library:
	-$(RM) ./Middlewares/USB_Device_Library/usbd_cdc.cyclo ./Middlewares/USB_Device_Library/usbd_cdc.d ./Middlewares/USB_Device_Library/usbd_cdc.o ./Middlewares/USB_Device_Library/usbd_cdc.su ./Middlewares/USB_Device_Library/usbd_core.cyclo ./Middlewares/USB_Device_Library/usbd_core.d ./Middlewares/USB_Device_Library/usbd_core.o ./Middlewares/USB_Device_Library/usbd_core.su ./Middlewares/USB_Device_Library/usbd_ctlreq.cyclo ./Middlewares/USB_Device_Library/usbd_ctlreq.d ./Middlewares/USB_Device_Library/usbd_ctlreq.o ./Middlewares/USB_Device_Library/usbd_ctlreq.su ./Middlewares/USB_Device_Library/usbd_ioreq.cyclo ./Middlewares/USB_Device_Library/usbd_ioreq.d ./Middlewares/USB_Device_Library/usbd_ioreq.o ./Middlewares/USB_Device_Library/usbd_ioreq.su

.PHONY: clean-Middlewares-2f-USB_Device_Library

