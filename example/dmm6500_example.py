from time import sleep
from lab_driver.driver import DriverDMM6500


if __name__ == "__main__":
    # Try connecting to device and reset it
    dmm = DriverDMM6500()
    dmm.serial_start()
    dmm.do_reset()
    
    # Measure AC voltage with 100 V range
    print("AC voltage")
    dmm.set_measurement_mode("VOLT", "AC")
    dmm.set_voltage_range(100, "AC")
    for i in range(5):
        print(dmm.get_voltage())
        sleep(.5)
    
    # Measure DC current with automatically selected range
    print("DC current")
    dmm.set_measurement_mode("CURR")
    dmm.set_current_range("AUTO")
    for i in range(5):
        print(dmm.get_current())
        sleep(.5)

    # Close the connection when we're done
    dmm.serial_close()
