import serial
import matplotlib
matplotlib.use("TkAgg")  # Force a common GUI backend
import matplotlib.pyplot as plt

def main():
    # Open the serial port at 115,200 baud
    ser = serial.Serial('COM6', 115200, timeout=1)
    
    # Set up the plot in interactive mode
    plt.ion()
    fig, ax = plt.subplots()
    data = []
    line_plot, = ax.plot([], [], label="Microphone Output")
    ax.set_xlabel("Sample #")
    ax.set_ylabel("Value")
    ax.legend()
    plt.show()  # Display the figure window

    max_values = 500  # We'll collect exactly 5000 samples

    while len(data) < max_values:
        line = ser.readline().decode('utf-8').strip()
        if line.startswith("Microphone_Output= "):
            try:
                value_str = line.split('=')[1].strip()
                value = int(value_str)
                print("Received:", value)
                data.append(value)

                # Update the plot
                line_plot.set_xdata(range(len(data)))
                line_plot.set_ydata(data)
                ax.relim()
                ax.autoscale_view()

                plt.draw()
                #plt.pause(0.01)  # Allow the GUI to update
            except ValueError:
                # Ignore lines that don't parse correctly
                pass

    # Close the serial port after collecting 5000 samples
    ser.close()
    print(f"Done collecting {max_values} values.")

    # Keep the plot open after data collection
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
