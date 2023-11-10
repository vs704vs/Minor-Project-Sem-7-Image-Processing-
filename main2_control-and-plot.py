import main2_sd
import main2_f1
import main2_f2
import main2_f3

print("Standard Day")
sd = main2_sd.sd()

print("F1")
f1 = main2_f1.f1()

print("F2")
f2 = main2_f2.f2()

print("F3")
f3 = main2_f3.f3()


print("Standard Day - " + str(sd))
print("F1 - " + str(f1))
print("F2 - " + str(f2))
print("F3 - " + str(f3))




# Import matplotlib.pyplot and numpy
import matplotlib.pyplot as plt
import numpy as np

# Create two arrays of data
x = np.array([1, 3, 5, 7, 9, 11, 13]) # x-axis values
y1 = np.array(sd) # y-axis values for the first line
y2 = np.array(f1) # y-axis values for the second line
y3 = np.array(f2) # y-axis values for the third line
y4 = np.array(f3) # y-axis values for the fourth line

# Plot the two arrays using different colors and labels
plt.plot(x, y1, color="black", label="Standard Day")
plt.plot(x, y2, color="blue", label="F1")
plt.plot(x, y3, color="red", label="F2")
plt.plot(x, y4, color="yellow", label="F3")

# Add a title and a legend to the plot
plt.title("Wound Ared Analysis")
plt.legend()

# Show the plot on the screen
plt.show()
