from sense_hat import SenseHat
import time

sense = SenseHat()
sense.clear()

# 색 정의
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

def show_message(message, color=WHITE, bg=(0, 0, 0), scroll_speed=0.08):
    sense.show_message(message, text_colour=color, back_colour=bg, scroll_speed=scroll_speed)

def show_entry():
    show_message("입차 허가", color=GREEN)

def show_exit():
    show_message("출차 허가", color=BLUE)

def show_denied():
    show_message("거부", color=RED)

def show_waiting():
    show_message("대기중", color=WHITE)

def clear_display():
    sense.clear() 