# driver_meas_devices
Python Driver for Measurement Devices from the Lab

## Installation for Debugging

We recommended to install all python packages for using this API with a virtual environment (venv). Therefore, we also recommend to `uv` ([Link](https://docs.astral.sh/uv/)) package manager. `uv` is not a standard package installed on your OS. For this, you have to install it in your Terminal (Powershell on Windows) with the following line.
````
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
````
Afterwards you can create the venv and installing all packages using this line.
````
uv venv
.\.venv\Scripts\activate  
uv sync
````

## Using the package release in other Repos
### MXO44 how-to on Linux:
1. Start RsVisa Tester
2. Click on Find Resource once after a reboot
3. Done

### DMM6500 how-to on Linux:
1. Don't yet

### NGU411 FastLog notes:
FastLog dumps a binary file of its measurements on the USB stick. This binary
file is entirely composed of single precision (32-bit) floating point numbers
in a pair format, that is, the first value is the voltage and the second value
is the current of the measurement. This pair format of measurements repeats
for the entire file. SI-Units are used (V and A). Values are stored in little
endian format. Conversion to CSV should be vastly faster using a simple C
program instead of the machine's built-in converter.
