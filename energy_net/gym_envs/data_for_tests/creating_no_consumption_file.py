from datetime import datetime, timedelta

start = datetime(2023, 12, 1, 0, 0)
step = timedelta(minutes=30)

with open("zero_consumption.csv", "w") as f:
    f.write("Datetime,Consumption\n")
    t = start
    for _ in range(10000):
        f.write(f"{t:%Y-%m-%d %H:%M:%S},0\n")
        t += step
