from elasticai.hw_measurements.driver import DriverRTM3004
from elasticai.hw_measurements import KHz
from time import sleep


if __name__ == '__main__':
    # Try connecting to the device and reset it
    rtm = DriverRTM3004()
    rtm.serial_start()
    rtm.do_reset()
    
    # Generate a sine wave with exponentially increasing frequency
    freq = 10 * KHz
    rtm.gen_function("SINE")
    rtm.gen_frequency(freq)
    rtm.gen_enable()
    for i in range(100):
        freq *= 1.05
        rtm.gen_frequency(freq)
        sleep(0.05)
    rtm.gen_disable()
    
    # Close the connection when we're done
    rtm.serial_close()
