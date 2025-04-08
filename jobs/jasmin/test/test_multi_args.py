import sys
import time

if __name__ == "__main__":
    print(sys.argv)
    name = sys.argv[1]
    age = int(sys.argv[2])
    height = float(sys.argv[3])
    sports = sys.argv[4].split(",")     # list
    wait = sys.argv[5].lower() in ("true", "1", "yes")
    if wait:
        total_time = 60  # seconds
        interval = 5  # seconds
        for i in range(0, total_time, interval):
            print(f"{i} seconds passed...")
            time.sleep(interval)
        print("Done after 1 minute!")

    print(f'Hello {name}, Age is {age}, Height is {height}. Age+Height is {age+height}\n'
          f'Sports is {sports}')