# driver_meas_devices
Python Driver for Measurement Devices from the Lab

MXO44 how-to on Linux:
1. Start RsVisa Tester
2. Click on Find Resource once after a reboot
3. Done

DMM6500 how-to on Linux:
1. Don't yet

NGU411 FastLog notes:

FastLog dumps a binary file of its measurements on the USB stick. This binary
file is entirely composed of single precision (32-bit) floating point numbers
in a pair format, that is, the first value is the voltage and the second value
is the current of the measurement. This pair format of measurements repeats
for the entire file. SI-Units are used (V and A). Values are stored in little
endian format. Conversion to CSV should be vastly faster using a simple C
program instead of the machine's built-in converter.
