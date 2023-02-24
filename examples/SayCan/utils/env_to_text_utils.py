'''
  Code taken from https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb
  Licensed under Apache 2.0 license
'''

import pybullet
import numpy as np

def build_scene_description(found_objects, block_name="box", bowl_name="circle"):
    scene_description = f"objects = {found_objects}"
    scene_description = scene_description.replace(block_name, "block")
    scene_description = scene_description.replace(bowl_name, "bowl")
    scene_description = scene_description.replace("'", "")
    return scene_description


def get_available_objects(env):
    return list(env.obj_name_to_id.keys())

def get_env_action_from_command(step, env, place_targets):
    step = step.replace("robot.pick_and_place(", "")
    step = step.replace(")", "")
    pick, place = step.split(", ")

    def _get_object_position(object_name):
        object_id = env.obj_name_to_id[object_name]
        object_pose = pybullet.getBasePositionAndOrientation(object_id)
        object_position = np.float32(object_pose[0])
        return object_position

    pick_position = _get_object_position(pick)
    if place in env.obj_name_to_id:
        place_position = _get_object_position(place)
    else:
        place_position = np.float32(place_targets[place])

    return {'pick': pick_position, 'place': place_position}


def get_possible_actions(pick_targets, place_targets, options_in_api_form=True, termination_string="done()"):
    options = []
    for pick in pick_targets:
        for place in place_targets:
            if options_in_api_form:
                option = "robot.pick_and_place({}, {})".format(pick, place)
            else:
                option = "Pick the {} and place it on the {}.".format(pick, place)
            options.append(option)

    options.append(termination_string)
    return options