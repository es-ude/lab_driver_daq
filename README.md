# Driver for DAQ Devices of the IES Lab
Python Driver for Measurement Devices from the Lab

## Using the package release in other Repos
Just adding this repo in the pyproject.toml file as dependencies by listing the git path.

## Installation
### Python for Developing
We recommended to install all python packages for using this API with a virtual environment (venv). Therefore, we also recommend to `uv` ([Link](https://docs.astral.sh/uv/)) package manager. `uv` is not a standard package installed on your OS. For this, you have to install it in your Terminal (Powershell on Windows) with the following line.
````
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
````
Afterwards with ``uv sync``, the venv will be created and all necessary packages will be installed.
````
uv venv
.\.venv\Scripts\activate  
uv sync
````
### Necessary drivers
For using the modules, it is necessary to install the VISA drivers for your OS in order to identify the USB devices, which will be known as IVI device. If you do not do this, then no access is available.
You can download it [here](https://www.rohde-schwarz.com/de/driver-pages/fernsteuerung/3-visa-und-tools_231388.html).

## How-To-Use
### MXO44 how-to on Linux:
1. Start RsVisa Tester
2. Click on Find Resource once after connecting via USB
3. Done

### DMM6500 how-to on Linux:
You need to do a first time setup
1. Check the USB connection `lsusb`
2. This file may be owned by root, so if that's the case:
`sudo chown <user> /sys/bus/usb/drivers/usbtmc/new_id`
3. Ensure usbtmc kernel module is loaded `sudo modprobe usbtmc`, then `ls /dev/usbtmc*`
and check if the device is listed
4. Create a udev rule to allow access to the device
`echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="05e6", ATTR{idProduct}=="6500", MODE="0666"' | sudo tee /etc/udev/rules.d/99-keithley.rules`
5. Reload the udev rules with `sudo udevadm control --reload-rules` and then `sudo udevadm trigger`
6. Restart the device
7. Crucial: *Open* pyvisa ResourceManager with @py backend instead of IVI-VISA, but let IVI
*scan* for devices beforehand!

## Notes / Errors for using the lab devices
### NGU411 FastLog
FastLog dumps a binary file of its measurements on the USB stick. This binary
file is entirely composed of single precision (32-bit) floating point numbers
in a pair format, that is, the first value is the voltage and the second value
is the current of the measurement. This pair format of measurements repeats
for the entire file. SI-Units are used (V and A). Values are stored in little
endian format. Conversion to CSV should be vastly faster using a simple C
program instead of the machine's built-in converter.

### MXO44
The device freezes upon the detection of a USB (dis-)connection event and will
cease all functionality on firmware version 2.4.2.1. The only fix is cutting
power to the device by holding the power button or plugging it from the socket.
