import keyboard
import time
from multiprocessing import Process, Value
from ctypes import c_bool

class KeyboardClient():
    def __init__(self):
        self.running = Value(c_bool, True, lock=True)

        self.upArrow = Value(c_bool, False, lock=True)
        self.downArrow = Value(c_bool, False, lock=True)
        self.leftArrow = Value(c_bool, False, lock=True)
        self.rightArrow = Value(c_bool, False, lock=True)

        self.process = Process(target=self.run)
        self.process.start()

    def run(self):
        while self.running.value:
            self.upArrow.value = keyboard.is_pressed('up')
            self.downArrow.value = keyboard.is_pressed('down')
            self.leftArrow.value = keyboard.is_pressed('left')
            self.rightArrow.value = keyboard.is_pressed('right')
            time.sleep(0.1)

    def stop(self):
        self.running.value = False
        self.process.join()

if __name__ == "__main__":
    kb = KeyboardClient()
    try:
        while True:
            print("Up = ", kb.upArrow.value, "; Down = ", kb.downArrow.value, "; Left = ", kb.leftArrow.value, "; Right = ", kb.rightArrow.value)
            time.sleep(1)
    except KeyboardInterrupt:
        kb.stop()
