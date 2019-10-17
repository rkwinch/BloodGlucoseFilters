import json # Used to parse JSON file
from scipy.signal import savgol_filter
import scipy.interpolate as interp
import numpy as np
import pylab
import matplotlib.pyplot as plt

# Transforms array yValues to be size n - See comments in main program
def CrunchArray(yValues, n):
    m = len(yValues)
    xValues = np.arange(m)
    # Make an interpolation function so we can approximate yValues at any x value
    interpolation_function = interp.interp1d(xValues, yValues) # from SciPy

    # Create an array from 0 to m-1 with n evenly spaced vamples
    # In other words, the array will have size n.
    # These are the x-values (counters) so as long as they are evenly spaced
    # it should plot nicely with how we did our other curves.
    new_x_values = np.linspace(0, m-1, n)
    return interpolation_function(new_x_values)

# Code taken from "Smoothing of a 1D Signal" from the SciPy Cookbook
def smooth(x,window_len,window):
  s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
  if window == 'flat': #moving average
    w=np.ones(window_len,'d')
  else:
    w=eval('np.'+window+'(window_len)')
  y=np.convolve(w/w.sum(),s,mode='valid')
  return y

def Plot(xvals, yvals, myLabel, raws):
    plt.plot(xvals, yvals, label=myLabel)
    plt.plot(xvals, raws, 'o',label="RawData",markersize=1)
    plt.legend()
    pylab.savefig(myLabel+'.png')
    plt.clf()
    
# Calculation of raw data derived from formula used in open source Nightscout project
# https://github.com/nightscout
def CalculateRaw(scale,intercept,slope,sgv,filtered,unfiltered):
  if sgv < 40:
    return scale * (unfiltered - intercept) / slope
  else:
    ratio = scale * (filtered - intercept) / slope / sgv
    return scale * (unfiltered - intercept) / slope / ratio

# Make a list of all the lines in our file
d = open("dump.json", "r").readlines()

# We want to search until we find a calibration point later
foundACalibration = False

calScale = 0
calIntercept = 0
calSlope = 0

sgvs = [] # List of Dexcom Displayed Glucose Values
raws = [] # List of Raw Glucose Values
dates = [] # Counter for number of data points
num_data_points = 0
for i in range(len(d)):
    line = d[i] #line of file
    jsonData = json.loads(line) # Parse the json into a dictionary
    typ = jsonData["type"]

    # The line with mbg immediately precedes the line with calibration always
    # The actual value on the mbg line is the given calibration from the user
    if foundACalibration == False and typ == "mbg":
      foundACalibration = True
    if foundACalibration == True:
      if typ == "mbg":
        nextLine = d[i+1]
        jsonDataOnNextLine = json.loads(nextLine) # line after mbg line (calibration line)

        # We need these values to calculate raw data points
        calScale = float(jsonDataOnNextLine["scale"])
        calIntercept = float(jsonDataOnNextLine["intercept"])
        calSlope = float(jsonDataOnNextLine["slope"])
      if typ == "sgv": # Blood Sugar Entry (Smoothed / Every 5 Minutes)
        device = jsonData["device"] # Filter out other devices (some are from the xDrip app which didn't work)
        if device == "dexcom":
          filtered = float(jsonData["filtered"])
          unfiltered = float(jsonData["unfiltered"])
          sgv = float(jsonData["sgv"]) # Dexcom Displayed/Smoothed Value
          raw = CalculateRaw(calScale, calIntercept, calSlope, sgv,filtered,unfiltered)
          sgvs.append(sgv)
          raws.append(raw)
          dates.append(num_data_points)
          num_data_points += 1
# Filter #1
# Savitzky-Golay Filter, Window Length 5, Order 2 Polynomial
# Savgol_Filter() returns filtered data, then we store it in a numpy array
savData = np.array(savgol_filter(raws, 5, 2, mode='nearest'))

# Filter #2
# Do a least squares fit
# Polyfit returns coefficients of a polynomial of degree d
# What degree to use? The data is quite jagged and noisy and not best
# approximated by low degree polynomials.  In retrospect, a
# least squares fit is probably not a great approach; however
# it is still included for demonstration purposes.
d = 10
lstSq = np.polyfit(dates,raws,d) # np.polyfit(x,y,degree)
# Now use the coefficients returned to make a polynomial function
lstSq = np.poly1d(lstSq) # After this, we can evaluate lstSq as a function.
lstSq = np.array(lstSq(dates)) # dates = data points 1-~1000 (x-values)

# Filters #3 and #4
# 1D Signal Processing Algorithms from SciPy Cookbook
flat = np.array(smooth(np.array(raws), 5, "flat")) # Average of raws per window (5)
hamming = np.array(smooth(np.array(raws), 5, "hamming")) # Reduces side peaks next to main peak

# One problem - these last 2 filters are the result of a convolution, which
# doesn't necessarily return a shape that will match our other filters.  We
# would like to "crunch" this data down to be the same size - essentially
# compress the data from size m to n.  We'll do this in two steps:
#     1) Create an interpolation function for these 2 filters to approximate their values
#        in the range [0, m] (their current size)
#     2) Use that interpolation function to populate a new array of size n
n = len(raws)
flat = CrunchArray(flat, n)
hamming = CrunchArray(hamming, n)

# Create combined plot of all filters
plt.plot(dates, sgvs, label='Dexcom_Value') # plt.plot(x, y, label=optional_label)
plt.plot(dates, raws, label='Raw_Value')
plt.plot(dates, savData, label='SavitkyGolay')
plt.plot(dates, lstSq, label='LeastSquares')
plt.plot(dates, flat, label='Flat')
plt.plot(dates, hamming, label='Hamming')
plt.legend()
pylab.savefig('combined.png')

# Clear the plot and now we will replot everything individually
plt.clf()

Plot(dates,sgvs,'Dexcom_Value', raws)
Plot(dates,raws,'Raw_Value', raws)
Plot(dates,savData,'SavitzkyGolay', raws)
Plot(dates,lstSq,'LeastSquares', raws)
Plot(dates,flat,'Flat', raws)
Plot(dates,hamming,'Hamming', raws)

# Now write the plot data out so we can do summary statistics in R
outputFile = open("PlotData.csv","w")
# Write header
outputFile.write("SGV, Raw, SavitzkyGolay, LeastSquares, Flat, Hamming\n")

# For each data point
for i in range(num_data_points):
  # For each filter
  for filterType in [sgvs, raws, savData, lstSq, flat, hamming]:
      # Write this filter's i'th data point and a comma
      outputFile.write(str(filterType[i]))
      # Don't write a comma if this is the last entry...
      if np.array_equal(filterType, hamming) == False:
        outputFile.write(',')
  outputFile.write('\n')

# Now, find which filter is closest to the dexcom value. (sgvs)
minimumDelta = 999999 # Initialize to large value 
winner = "No winner"
for filterType in [savData, lstSq, flat, hamming]:
    dexcom = np.array(sgvs)
    absDiff = np.absolute(dexcom - filterType)
    avgDelta = np.mean(absDiff)

    if avgDelta < minimumDelta:
        minimumDelta = avgDelta
        # Not sure why np.array_equal() is needed here instead of ==
        if np.array_equal(filterType,lstSq):
            winner = "Least Squares"
        elif np.array_equal(filterType,savData):
            winner = "Savitzky-Golay"
        elif np.array_equal(filterType,flat):
            winner = "Flat"
        elif np.array_equal(filterType,hamming):
            winner = "Hamming"
print "The best filter match is ", winner
