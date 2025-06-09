from time import sleep
from lab_driver import DriverMXO4X
from lab_driver.units import KHz, MHz


if __name__ == '__main__':
    # Try connecting to the device, reset it and display the waveform GUI with a sample text
    mx = DriverMXO4X()
    mx.serial_start()
    mx.do_reset()
    mx.set_display_activation(True)
    mx.set_static_display_text("Hello World!")
    
    # Generate a sine wave with exponentially increasing frequency
    freq = 10 * KHz
    mx.gen_function("SINE")
    mx.gen_frequency(freq)
    mx.gen_enable()
    for i in range(100):
        freq *= 1.08
        mx.gen_frequency(freq)
        sleep(0.05)
    mx.gen_disable()
    
    # Enter Frequency Response Analysis mode, which unlocks all fra_* functions for use.
    # This is done automatically whenever you call an fra_* function outside of FRA mode.
    mx.fra_enter()
    mx.fra_freq_start(10*KHz)
    mx.fra_freq_stop(1*MHz)
    print("Running FRA...")
    mx.fra_run()
    mx.fra_wait_for_finish()
    # mx.fra_stop()     to stop FRA manually
    print("FRA done.")
    
    # Close the connection when we're done
    mx.serial_close()
