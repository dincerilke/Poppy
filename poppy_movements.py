#
# poppy_movements.py - file for testing vision
# Project - makerspace Poppy robot
#
# Author: Ilke Dincer
# Revisions:        initial version
#

from pypot.creatures import PoppyHumanoid

poppy = PoppyHumanoid()

[p.name for p in poppy.primitives]

def wave():
    pass

def turn_left():
    # poppy.head_z.goal_position = 5
    pass

def turn_right():
    pass