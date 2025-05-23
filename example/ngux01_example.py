from lab_driver import DriverNGUX01


if __name__ == '__main__':
    # Try connecting to the device and reset it
    ngu = DriverNGUX01()
    ngu.serial_start()
    ngu.do_reset()
    
    # Set 50k samples per second and 6 seconds of runtime for FastLog, then run it. Beep twice when finished
    ngu.set_fastlog_sample_rate(50)
    ngu.set_fastlog_duration(6)
    print("Searching for USB device...")
    ngu.event_handler(ngu.is_usb_connected, ngu.do_fastlog)
    print("Found USB device. Running FastLog...")
    ngu.event_handler(ngu.is_fastlog_finished, ngu.do_beep, 2)
    print("Done.")
    
    # Close the connection when we're done
    ngu.serial_close()
