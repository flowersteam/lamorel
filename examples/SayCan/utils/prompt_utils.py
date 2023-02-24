'''
  Code taken from https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb
  Licensed under Apache 2.0 license
'''

def get_in_context_examples():
    return \
"""
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue
block, top
left
corner)
robot.pick_and_place(red
block, top
left
corner)
robot.pick_and_place(yellow
block, top
left
corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one the green thing.
robot.pick_and_place(yellow
block, green
bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow
block, middle)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green
block, blue
block)
robot.pick_and_place(red
block, green
block)
done()

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue
block, blue
bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green
block, green
bowl)
robot.pick_and_place(red
block, red
bowl)
robot.pick_and_place(yellow
block, yellow
bowl)
done()
"""