import serial
import numpy as np

class com_usb():
    """Class for handling serial ports in Python"""
    def __init__(self, com_name: str, baud: int):
        """Init. of the device with name and baudrate of the device"""
        self.SerialName = com_name
        self.SerialBaud = baud
        self.SerialParity = 0

        self.device = serial.Serial()
        self.device_init = False

    def setup_usb(self):
        """Setup USB device"""
        # Setting the parity
        parity = str()
        if self.SerialParity == 0:
            parity = serial.PARITY_NONE
        if self.SerialParity == 1:
            parity = serial.PARITY_EVEN
        if self.SerialParity == 2:
            parity = serial.PARITY_ODD

        # Setting the serial port
        self.device = serial.Serial(
            port=self.SerialName,
            baudrate=self.SerialBaud,
            parity=parity,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
        self.device_init = True

    def open(self):
        """Starting a connection to device"""
        if self.device.is_open:
            self.device.close()
        self.device.open()

    def close(self):
        """Closing a connection to device"""
        self.device.close()

    def write_wofb(self, data: bytes) -> None:
        """Write content to device without feedback"""
        self.device.write(data)

    def write_wfb(self, data: bytes):
        """Write all information to device (specific bytes)"""
        bytes_size = len(data)
        self.device.write(data)
        dev_out = self.device.read(bytes_size)
        return dev_out

    def write_wfb_lf(self, data: bytes) -> bytes:
        """Write all information to device (unlimited bytes until LF)"""
        self.device.write(data)
        dev_out = self.device.read_until()
        return dev_out

    def write_wfb_hex(self, head: int, data: int) -> int:
        """Write content to FPGA/MCU for specific custom-made task"""
        data0 = int(data)

        transmit_byte = self.__process_data_to_dev(head, data0)
        bytes_size = len(transmit_byte)

        self.device.write(transmit_byte)
        dev_out = self.device.read(bytes_size)
        out = self.__process_data_from_dev(1, dev_out)

        return out

    def read(self, no_bytes: int):
        """Read content from device"""
        return self.device.read(no_bytes)

    def process_to_dev(self, head: int, data: int) -> bytes:
        data0 = int(data)
        out = head.to_bytes(1, 'little')
        out += data0.to_bytes(2, 'big')
        return out

    def process_from_dev(self, data: bytes) -> list:
        out = []
        packet_size = 3
        no_iteration = int(len(data)/packet_size)
        for idx in range(no_iteration):
            val = data[idx*packet_size : (idx+1)*packet_size]
            out.append(self.__process_data_from_dev(1, val))
        return out

    def __process_data_to_dev(self, head: int, data: int) -> bytes:
        out = head.to_bytes(1, 'little')
        out += data.to_bytes(2, 'big')
        return out

    def __process_data_from_dev(self, head_num: int, xin: bytes) -> int:
        used = xin[head_num:]
        out = int.from_bytes(used, byteorder='big')
        return out





