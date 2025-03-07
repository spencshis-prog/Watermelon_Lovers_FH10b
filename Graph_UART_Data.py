import serial
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt

def main():
    ser = serial.Serial('COM6', 115200, timeout=1)
    
    
    plt.ion()
    fig, ax = plt.subplots()
    data = []
    line_plot, = ax.plot([], [], label="Microphone Output")
    ax.set_xlabel("Sample #")
    ax.set_ylabel("Value")
    ax.legend()
    plt.show() 

    discard_count = 1500  
    max_values = 500     

    while len(data) < max_values:
        line = ser.readline().decode('utf-8').strip()
        
        try:
            if "Microphone_Output=" in line:
                value_str = line.split('=')[1].strip()
                value = int(value_str)
            else:
                value = int(line)
                
            print("Received:", value)
            
           
            if discard_count > 0:
                discard_count -= 1
                continue

           
            data.append(value)
            
           
            line_plot.set_xdata(range(len(data)))
            line_plot.set_ydata(data)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
           
        except ValueError:
            
            pass

   
    ser.close()
    print(f"Done collecting {max_values} values.")


    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
