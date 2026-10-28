# Low-Power Data Logger with LPWAN Communication

*Data logger de bajo consumo con comunicación LPWAN*

This repository contains the hardware, firmware and design-support tools for a family of low-power environmental data loggers (weather stations) built around the **STM32L476RG** microcontroller. The loggers measure temperature, relative humidity, barometric pressure and rainfall, store every sample on a microSD card, and report averaged readings over an LPWAN link (**LoRaWAN** or **NB-IoT**). Each radio option is available in a battery-only and a solar-harvesting version.

The repository is organised in three parts:

1. **Four KiCad PCB projects** - one per radio/power combination.
2. **Firmware for the LoRa Solar board** - an STM32CubeIDE project in `test/`.
3. **A board selection tool** - a Python/Streamlit application that recommends the most suitable board for a given site in Chile, together with the solar and cellular-coverage data it needs.

---

## Repository structure

```
.
├── MCU_lora.zip                  KiCad project - LoRa, battery-powered
├── MCU_lora_solar.zip            KiCad project - LoRa, solar-harvesting (tested)
├── MCU_Nb_IoT.zip                KiCad project - NB-IoT, battery-powered
├── MCU_Nb_IoT_solar.zip          KiCad project - NB-IoT, solar-harvesting
├── test/                         STM32CubeIDE firmware for the LoRa Solar board
├── herramienta_de_selecccion/    Board selection tool (Python / Streamlit)
│   ├── selectorv2.py             Application
│   ├── boards.json               Board registry (power profiles, capabilities)
│   ├── solar_data.zip            Solar irradiance and temperature rasters (must be extracted)
│   └── Subtel/                   Cellular base-station datasets (March 2025)
├── LICENSE
└── README.md
```

---

## 1. Hardware - PCB projects

All four boards are **4-layer KiCad 9 projects** and share the same core architecture. Each `.zip` file contains a complete, self-contained KiCad project (hierarchical schematic, PCB layout, design rules and project-local library tables). Extract the archive and open the `.kicad_pro` file with **KiCad 9.0 or later**.

### Common core (all variants)

| Function | Component |
|---|---|
| Microcontroller | STMicroelectronics STM32L476RGT6 (Cortex-M4F, LQFP64) |
| Air temperature | Texas Instruments TMP117 (±0.1 °C digital sensor) |
| Humidity / pressure | Bosch BME280 |
| Real-time clock and wake-up timer | Micro Crystal RV-3028-C7 (ultra-low-power RTC) |
| Local storage | microSD card over SPI (FatFs) |
| Main regulator | Texas Instruments TPS62840 (ultra-low-Iq buck converter) |
| Switched power rails | Texas Instruments TPS22916 load switches (sensor rail and transceiver rail) |
| USB | USB-C port with USBLC6-2 ESD protection (USB-CDC console and power) |
| Rain gauge | Tipping-bucket input on an interrupt-capable GPIO |

Peripherals are grouped on two load-switched rails (`V_SENSOR` and `V_TRANS`) so that sensors, the SD card and the radio can be completely de-energised while the MCU sleeps.

### Variant summary

| Project | Radio | Power source | Notes |
|---|---|---|---|
| `MCU_lora` | Fanstel LR62E (Semtech SX1262), LoRaWAN | Primary battery (e.g. Li-SOCl₂) | |
| `MCU_lora_solar` | Fanstel LR62E (Semtech SX1262), LoRaWAN | Solar panel + rechargeable cell (LiPo / Li-ion) | **Built and tested.** Includes schematic PDF and STEP 3D model. |
| `MCU_Nb_IoT` | Quectel BC660K-GL, NB-IoT | Primary battery (e.g. Li-SOCl₂) | Includes SIM holder and U.FL antenna connector. |
| `MCU_Nb_IoT_solar` | Quectel BC660K-GL, NB-IoT | Solar panel + rechargeable cell (LiPo / Li-ion) | |

The **solar** variants add a Texas Instruments **BQ25570** energy-harvesting charger (MPPT boost charger with programmable over/under-voltage thresholds) to the `Power` sheet. Its `VBAT_OK` and `VOUT_EN` signals are routed to the MCU.

The NB-IoT archives also include the Quectel reference documents used during the design (`nbiot/`): hardware design guide, AT command manual and low-power design guide for the BC660K-GL.

### Schematic sheets

Each project is split into the following hierarchical sheets:

| Sheet | Content |
|---|---|
| `Power.kicad_sch` | Battery input, buck regulator, load switches (and BQ25570 in solar variants) |
| `LoRa_RF.kicad_sch` / `BC660K_GL.kicad_sch` | Radio module, RF path and antenna |
| `BME280.kicad_sch`, `TMP117.kicad_sch` | Environmental sensors |
| `RTC_RV-3028-C7.kicad_sch` | External RTC |
| `MicroSD.kicad_sch` | SD card socket |
| `USBC_port.kicad_sch` | USB-C connector and protection |

---

## 2. Firmware - LoRa Solar board (`test/`)

The `test/` folder contains the firmware that runs on the **MCU_lora_solar** board. It is an **STM32CubeIDE** project generated with STM32CubeMX 6.16.1 and the **STM32Cube FW_L4 V1.18.2** package.

### Main features

- **LoRaWAN 1.0.4 stack** based on Semtech *LoRaMac-node*, ported to this board (`Middlewares_node/lora/`), using the SX126x radio driver for the LR62E module.
- **Region AU915**, ABP activation, data rate DR6 (SF8 / 500 kHz) on the single 500 kHz channel 64 (915.9 MHz); ADR disabled; unconfirmed uplinks.
- **Ultra-low-power operation**: the MCU stays in STOP mode and is woken by the RV-3028 periodic timer. On each wake-up it powers the sensor rail, takes one sample, writes it to the SD card, and powers everything down again.
- **Averaged uplinks**: every *N* wake-ups, the firmware reads the last *N* samples from the SD card, averages them, and transmits a 7-byte payload. The radio is powered only for the transmission.
- **Rain gauge**: tipping-bucket pulses are counted by interrupt; the uplink reports the cumulative count since local midnight.
- **Frame-counter persistence**: the uplink frame counter is stored in an RTC backup register, so it survives sleep cycles and soft resets.
- **USB serial console (microshell)**: when USB is connected, a command shell is available on the virtual COM port.

### Project layout

| Path | Content |
|---|---|
| `Core/Src/main.c` | Initialisation, sleep/wake loop, microshell hooks |
| `Core/Src/functions.c` | Sampling, SD logging, payload encoding, LoRaWAN callbacks, power-rail control |
| `Middlewares_node/Src/` | Drivers: `bme280.c`, `tmp117.c`, `rv3028.c`, `user_diskio_spi.c` (SD over SPI), `microshell.c` |
| `Middlewares_node/lora/` | LoRaMac-node stack: MAC, AU915 region, SX126x radio, board port, software secure element |
| `FATFS/`, `USB_DEVICE/` | CubeMX-generated FatFs and USB-CDC middleware |
| `test.ioc` | STM32CubeMX configuration (pin-out and clock tree) |

### Building and flashing

1. Install **STM32CubeIDE** (with the STM32Cube FW_L4 V1.18.2 package) and **STM32CubeProgrammer**.
2. *File → Import → Existing Projects into Workspace* and select the `test/` folder.
3. Build the **Release** configuration. The **Debug** configuration is intended only for development.
4. For development and debugging, connect an ST-LINK to the SWD pins (PA13 / PA14) and flash using the included `test Debug.launch` configuration.
5. **For deployment and low-power measurements, program the board with STM32CubeProgrammer instead of the IDE debugger:**
   1. Open STM32CubeProgrammer and connect to the board through the ST-LINK (SWD).
   2. Load the Release output file (`test/Release/test.elf` or `.hex`) and program the device.
   3. Disconnect the ST-LINK from the board.
   4. Power-cycle the board completely by removing and reconnecting the battery and USB.

> **Why this matters:** A board flashed or started from a debug session in STM32CubeIDE keeps the MCU debug logic active. This also happens if the ST-LINK stays connected. Clocks then keep running in STOP mode, and the sleep current rises well above the expected value. Programming with STM32CubeProgrammer, disconnecting the probe and doing a full power cycle turns off all debug-related consumption. Only then do current measurements show the board's real low-power performance.

### LoRaWAN provisioning

> **Important:** The LoRaWAN identifiers and keys in this repository have been removed and replaced with **all-zero placeholders**. The firmware will build, but it will not communicate with any network until real credentials are entered.

The device uses **ABP** activation. Before building firmware for a board, register an end device on your network server (e.g. The Things Network, AU915 with channel 64 enabled) and replace the placeholder values in:

```
test/Middlewares_node/lora/Crypto/se-identity.h
    LORAWAN_DEVICE_EUI        Device EUI
    LORAWAN_JOIN_EUI          Join (application) EUI
    LORAWAN_DEVICE_ADDRESS    DevAddr (ABP)
    NWK_S_ENC_KEY, F_NWK_S_INT_KEY, S_NWK_S_INT_KEY   Network session key (ABP)
    APP_S_KEY                 Application session key (ABP)
    APP_KEY, NWK_KEY          Root keys (only needed for OTAA)
```

The region, uplink port and channel plan are set in `Core/Inc/functions.h` and `ApplyAu915SubBand1Mask()` in `Core/Src/functions.c`.

Every device must have its own DevAddr and keys. Keep them in a local, untracked copy of `se-identity.h`.

### Pin assignment (main signals)

| Signal | Pin | Signal | Pin |
|---|---|---|---|
| Radio SPI (SPI1) SCK / MISO / MOSI | PA5 / PA6 / PA7 | SD SPI (SPI2) SCK / MISO / MOSI | PB10 / PC2 / PB15 |
| Radio NSS / RESET | PC4 / PC5 | SD chip select | PB11 |
| Radio BUSY / DIO1 | PB1 / PB0 | Sensor I²C (I2C2) | BME280, TMP117, RV-3028 |
| Radio antenna switch | PB2 | RTC interrupt | PA8 |
| `V_SENSOR` enable | PA2 | Rain gauge input | PC13 |
| `V_TRANS` enable | PA1 | USB detect / wake-up | PB4 / PB6 |
| BQ25570 VBAT_OK / VOUT_EN | PA15 / PC10 | SWD | PA13 / PA14 |

### Uplink payload (7 bytes, port 2, big-endian)

| Bytes | Field | Type | Scaling |
|---|---|---|---|
| 0-1 | Rain pulses since midnight | `uint16` | pulses |
| 2-3 | Temperature (TMP117, averaged) | `int16` | °C × 100 |
| 4 | Relative humidity (BME280, averaged) | `uint8` | % |
| 5-6 | Pressure (BME280, averaged) | `uint16` | hPa × 10 |

No timestamp is transmitted; the network server provides the reception time. A payload formatter for The Things Network:

```javascript
function decodeUplink(input) {
  var b = input.bytes;
  var t = (b[2] << 8) | b[3];
  if (t & 0x8000) t -= 0x10000;
  return {
    data: {
      rain_pulses: (b[0] << 8) | b[1],
      temperature_c: t / 100,
      humidity_pct: b[4],
      pressure_hpa: ((b[5] << 8) | b[6]) / 10
    }
  };
}
```

### Data stored on the SD card

- `DATA.csv` - one line per sample: rain pulse count, temperature (°C), humidity (%RH), pressure (hPa), time and date.
- `DEBUG.txt` - debug messages.

### USB console commands

Connect the board over USB and open the virtual COM port with any serial terminal.

| Command | Description |
|---|---|
| `help` | List available commands |
| `rtc get` / `rtc set YYYY-MM-DD HH:MM:SS` | Read or set the external RTC |
| `wake get` / `wake set <seconds>` | Read or set the sampling (wake-up) interval. Values of 60 s or more are rounded up to whole minutes. |
| `txint get` / `txint set <seconds>` | Read or set the uplink interval (converted to a whole number of wake-ups) |
| `status` | Show sample counter, TX divider, wake interval and radio/USB state |
| `txtest` | Power the radio and send a fixed test uplink |
| `regs` | Read SX1262 diagnostic registers (DIO3/TCXO, crystal trim, RX gain) |

Default settings: one sample every 60 s, one uplink per wake-up.

### Known limitations and recommended improvements

The current firmware was adapted to work around the behaviour of the Fanstel LR62E module on this board. As a result, the LoRaWAN implementation is **simplified and does not fully comply with the LoRaWAN specification**:

- Only one uplink channel is used (500 kHz channel 64), instead of the regional AU915 channel plan.
- Transmissions are "fire-and-forget": after each uplink the MAC timers are stopped and the receive windows (RX1/RX2) are not opened, so the device cannot receive downlinks, MAC commands or acknowledgements.
- Adaptive data rate (ADR) and duty-cycle control are disabled, and the data rate is fixed at DR6.
- Activation is ABP only; OTAA join is not used.

For future revisions, the following is **recommended**:

1. **Replace the LR62E with a simpler, generic SX1262-based LoRa module** that exposes the standard radio interface (SPI, BUSY, DIO1, NRESET, and documented TCXO and RF-switch control), so that it works with the unmodified Semtech reference board support. This change affects the `LoRa_RF` sheet and the PCB layout of the two LoRa projects.
2. **Reincorporate the complete LoRaWAN stack from Semtech LoRaMac-node** (`LmHandler` with its standard state machine) instead of the current workarounds, so that the device complies with the LoRaWAN standard: OTAA activation, full AU915 sub-band channel plan, RX1/RX2 receive windows, ADR, duty-cycle handling and MAC command processing. The low-power sleep/wake structure of the current firmware can be kept; only the radio sequence needs to let the MAC finish each transaction before the radio is powered down.

---

## 3. Board selection tool (`herramienta_de_selecccion/`)

`selectorv2.py` is a **Streamlit** web application that recommends which of the boards above is best suited for a given installation site in Chile. For a chosen latitude/longitude it:

1. Looks up the **long-term solar irradiance (GHI)** and **average air temperature** at the site.
2. Checks **cellular coverage** (4G / LTE-M / NB-IoT) from the distance to the nearest registered base station.
3. Estimates each board's **daily energy consumption** from its power profile and the selected sampling rate.
4. Estimates the **solar energy yield** for both the annual average and a **winter-solstice worst case** (sun-angle reduction plus a latitude-based cloudiness penalty for southern Chile, following the sizing criterion of IEEE 1562).
5. Ranks the boards and shows the top recommendation, a ranked table and a map with nearby base stations.

A second tab, **Board Manager**, allows new board designs to be added to `boards.json` from the interface.

### Requirements

- Python 3.10 or later
- Packages:

```bash
pip install streamlit streamlit-folium folium geopandas shapely rasterio pandas numpy pyarrow
```

### Data setup

1. **Solar data** - extract `solar_data.zip` inside `herramienta_de_selecccion/` so that the following files exist:

   ```
   herramienta_de_selecccion/solar_data/GHI.asc
   herramienta_de_selecccion/solar_data/TEMP.asc
   ```

   These are ESRI ASCII grid rasters for Chile from the **Global Solar Atlas** (GHI in kWh/m²/day, air temperature in °C). Note that the extracted files take about 1.1 GB. If they are missing, the tool falls back to default values (GHI = 5.0 kWh/m²/day, 15 °C).

2. **Cellular data** - the `Subtel/` folder already contains the base-station registry published by the Chilean telecommunications authority (**SUBTEL**), period **March 2025**, for Entel, Movistar, Claro and WOM, by technology (2G, 3G, 4G, 5G). Each dataset is provided as GeoJSON (`.json`) and as GeoParquet (`.parquet`); the tool loads the Parquet file first because it is much faster. The expected naming pattern is `<Operator>_<Tech>_marzo2025.<ext>`.

### Running the tool

```bash
cd herramienta_de_selecccion
streamlit run selectorv2.py
```

The application opens in the browser. In the sidebar, set:

- latitude and longitude (limited to continental Chile),
- shading condition (full sun, partial shade, under canopy),
- samples per hour,
- required communication links,
- battery-only mode (no solar), and
- preferred mobile operator (optional),

and then click **Recommend**.

### Board registry (`boards.json`)

Each entry describes one board design:

| Field | Meaning |
|---|---|
| `key`, `name` | Unique identifier and display name |
| `sensors` | Measured variables (e.g. `temp`, `rh`, `pressure`, `rain_tipping`) |
| `comms` | Supported links (`LoRa`, `NB-IoT`, `LTE-M`, …) |
| `solar_ok`, `non_solar_ok` | Whether the board supports solar and/or battery-only operation |
| `battery_chemistries` | Supported cells (e.g. `LiPo`, `Li-Ion`, `Li-SOCl2`) |
| `max_panel_Wp` | Largest solar panel the board is designed for (Wp) |
| `notes` | Free text shown with the recommendation |
| `github` | Optional link to the design files (shown as "View design files") |
| `power` | Power profile: sleep current (µA), sampling current (mA) and duration (ms), transmit current (mA) and duration (s), transmit interval (min), and constant system overhead (mW) |

The four boards in this repository are already registered. The power profiles can be updated with measured values as boards are characterised.

### Model assumptions and limitations

- Cellular coverage is approximated by distance to the nearest base station (4G within 15 km; NB-IoT assumed only on Entel 4G sites within 25 km). It does not model terrain or signal strength.
- LoRa gateway availability is not yet evaluated (the check always returns *available*).
- Battery-only boards are accepted when their estimated consumption is at or below 150 mWh/day.
- A solar board is accepted only if the winter yield covers the load with a 10 % margin **and** the annual-average yield covers it with a 25 % margin.

These results are intended as a first-pass sizing aid and should be confirmed with on-site measurements.

---

## Data sources

- **Global Solar Atlas** (World Bank Group / Solargis) - GHI and air-temperature rasters.
- **SUBTEL**, Subsecretaría de Telecomunicaciones de Chile - mobile base-station registry, March 2025.
- **Semtech LoRaMac-node** - LoRaWAN stack (Revised BSD License).
- **STMicroelectronics STM32Cube FW_L4** - HAL, USB device and FatFs middleware (licensed under their respective terms).

## Author

Daisy Alexandra Berríos Araya - Ingeniería Civil Electrónica, Universidad Técnica Federico Santa María (UTFSM).

## License

This project is released under the [MIT License](LICENSE). Third-party components keep their original licenses.
