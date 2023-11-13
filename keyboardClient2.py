import curses
from multiprocessing import Process, Value
from ctypes import c_bool
import time

class KeyboardClient():
    def __init__(self):
        self.running = Value(c_bool, True, lock=True)
        self.upArrow = Value(c_bool, False, lock=True)
        self.downArrow = Value(c_bool, False, lock=True)
        self.leftArrow = Value(c_bool, False, lock=True)
        self.rightArrow = Value(c_bool, False, lock=True)

        args = (self.running, self.upArrow, self.downArrow, self.leftArrow, self.rightArrow)
        self.process = Process(target=self.run, args=args)
        self.process.start()

    def run(self, running, upArrow, downArrow, leftArrow, rightArrow):
        def process_input(stdscr):
            curses.cbreak()
            stdscr.keypad(True)
            stdscr.nodelay(True)
            while running.value:
                key = stdscr.getch()
                if key == curses.KEY_UP:
                    upArrow.value = True
                elif key == curses.KEY_DOWN:
                    downArrow.value = True
                elif key == curses.KEY_LEFT:
                    leftArrow.value = True
                elif key == curses.KEY_RIGHT:
                    rightArrow.value = True
                else:
                    upArrow.value = False
                    downArrow.value = False
                    leftArrow.value = False
                    rightArrow.value = False
                time.sleep(0.1)

        curses.wrapper(process_input)

    def stop(self):
        self.running.value = False
        self.process.join()

if __name__ == "__main__":
    kb = KeyboardClient()
    try:
        while True:
            print("Up Arrow: ", kb.upArrow.value, "; Down Arrow: ", kb.downArrow.value, 
                  "; Left Arrow: ", kb.leftArrow.value, "; Right Arrow: ", kb.rightArrow.value)
            time.sleep(0.1)
    finally:
        kb.stop()
