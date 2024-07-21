import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

class TIME:
    nanosecond = 1
    second = nanosecond * 10**9
    minute = second * 60
    hour = minute * 60
    day = hour * 24


def mean_fn(l):
    return sum(l) / len(l)

temp_dataset = pd.read_csv("data/temperature.csv") # https://mesonet.agron.iastate.edu/request/download.phtml?network=OR_ASOS
# Load hourly temperature data from csv


temp_dataset["valid"] = pd.to_datetime(temp_dataset.valid, format="%Y-%m-%d %H:%M", utc=True) # Convert str datatime to dt object
temp_dataset["valid"] = temp_dataset["valid"].apply(lambda x: x.replace(minute=0, second=0, microsecond=0)) # Round temperature time to nearest hour

def get_temp(epoch_time): # Finds the temperature for a given time from main dataset
    epoch_time = epoch_time / (TIME.second) # nanoseconds to seconds
    dt = datetime.fromtimestamp(epoch_time)
    dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=dt.minute//30) # round to nearest hour

    t = temp_dataset[temp_dataset.valid == dt.isoformat()].iloc[0].tmpf # find corresponding temp
    #print(dt, t)
    return t

data = pd.read_csv("data/SouthPole/L3.csv") # Leg 3 (Only compressor)
data = data[["time", "value"]]

data["time"] = data.time.map(lambda x: x + (8 * TIME.hour)) # Convert OurTable times to GMT


data = data[data.time < 1718983560000000000] # Friday, June 21, 2024 3:26:00 PM GMT
data = data[data.time > 1717085280000000000] # Thursday, May 30, 2024 4:08:00 PM GMT

#data["temp"] = data.time.apply(get_temp) # Get temperature values for time frame


start_time = int(data.iloc[0].time)
end_time = int(data.iloc[-1].time)


data_period = 10 * TIME.second # Our Table date polling rate

def get(t):
    if t in data.time.values:
        return data[data.time == t].iloc[0].value
    else:
        return get(t + data_period)


new_times = []
new_values = []
for t in range(start_time, end_time, data_period):
    if t in data.time.values:
        pass
    else:
        new_times.append(t)
        new_values.append(data[data.time == (data.time - data_period)])




BLINE = "black"
COMP = "green"
PEAK = "red"


def state(value): # These are arbitrary values... TODO?
    if value < 500:
        return BLINE
    elif value < 700:
        return COMP
    else:
        return PEAK

output = []

all_stacks = []


def iterate_data():
    last_state = BLINE
    curr_state = BLINE
    stack = []

    for time in tqdm(data.time):
        last_state = curr_state

        value = get(time)
        curr_state = state(value)

        if curr_state != BLINE: #starting/continuing cycle
                stack.append(
                    {"time": time, "value": value, "state": curr_state})

        if curr_state == BLINE and last_state != BLINE: # ending cycle
                if PEAK not in [i["state"] for i in stack]:
                    all_stacks.append(stack)

                stack = []

iterate_data()



x = []
y = []
temp = []

for stack in all_stacks:
    duration = stack[-1]["time"] - stack[0]["time"]

    midtime = stack[-1]["time"]

    x.append(midtime)
    y.append(duration)
    #temp.append(data[data.time == midtime].iloc[0].temp)
    temp.append(get_temp(midtime))


#plt.scatter(data.time, data.value, color=color)
#tt = 1.71847642 * 10**18
#plt.show()

x = np.array(x)
y = np.array(y)
temp = np.array(temp)


plt.figure(1)

plt.scatter(x, y, color="black")
plt.plot(x, temp * mean_fn(y) / mean_fn(temp), color="blue")

#plt.plot(data.time, data.temp * mean_fn(y) / mean_fn(data.temp), color="blue")
plt.show(block=False)



plt.figure(2)

m, b = np.polyfit(temp, y, 1)

wattage_hat = m * temp + b * (temp / temp)

resid = y - wattage_hat
resid = resid / mean_fn(abs(resid))


plt.scatter(x, resid)
plt.show()