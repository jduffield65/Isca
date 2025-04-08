import sys
import time
from typing import List

def main(name: str, age: int, height: float, sports: List, wait: bool):
    if wait:
        total_time = 60  # seconds
        interval = 5  # seconds
        for i in range(0, total_time, interval):
            print(f"{i} seconds passed...")
            time.sleep(interval)
        print("Done after 1 minute!")

    print(f'Hello {name}, Age is {age}, Height is {height}. Age+Height is {age+height}\n'
          f'Sports is {sports}')

if __name__ == "__main__":
    # print(sys.argv)
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4].split(","),
         sys.argv[5].lower() in ("true", "1", "yes"))
