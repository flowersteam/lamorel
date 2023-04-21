'''
  Code taken from https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb
  Licensed under Apache 2.0 license
'''

from environment import PickPlaceEnv, PICK_TARGETS, PLACE_TARGETS
from utils.env_to_text_utils import get_possible_actions, build_scene_description, get_env_action_from_command, get_available_objects
from utils.prompt_utils import get_in_context_examples
from utils.plot_utils import plot_saycan
from utils import normalize_scores
import hydra
import numpy as np
import matplotlib.pyplot as plt

from lamorel import Caller, lamorel_init
lamorel_init()


def affordance_scoring(options, found_objects, verbose=False, block_name="box", bowl_name="circle",
                       termination_string="done()"):
    '''
    Given this environment does not have RL-trained policies or an asscociated value function, we use affordances through an object detector.
    '''
    affordance_scores = {}
    found_objects = [
        found_object.replace(block_name, "block").replace(bowl_name, "bowl")
        for found_object in found_objects + list(PLACE_TARGETS.keys())[-5:]]
    verbose and print("found_objects", found_objects)
    for option in options:
        if option == termination_string:
            affordance_scores[option] = 0.2
            continue
        pick, place = option.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
        affordance = 0
        found_objects_copy = found_objects.copy()
        if pick in found_objects_copy:
            found_objects_copy.remove(pick)
            if place in found_objects_copy:
                affordance = 1
        affordance_scores[option] = affordance
        verbose and print(affordance, '\t', option)
    return affordance_scores


def plot_scene(env):
    plt.imshow(env.get_camera_image())
    plt.show()


@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # Instantiate Lamorel's Caller
    lm_server = Caller(config_args.lamorel_args)

    # Set environment
    env = PickPlaceEnv()
    np.random.seed(2)
    possible_actions = get_possible_actions(PICK_TARGETS, PLACE_TARGETS, config_args.rl_script_args.termination_str)
    task = "put all the blocks in different corners."
    base_prompt = get_in_context_examples()

    for i in range(config_args.rl_script_args.n_tasks):
        print(f"Task {i}")
        # Generate current scene
        pick_items = list(PICK_TARGETS.keys())
        pick_items = np.random.choice(pick_items, size=np.random.randint(1, 5), replace=False)
        place_items = list(PLACE_TARGETS.keys())[:-9]
        place_items = np.random.choice(place_items, size=np.random.randint(1, 6 - len(pick_items)), replace=False)
        config = {"pick": pick_items,
                  "place": place_items}
        env.reset(config)
        plot_scene(env)
        available_objects = get_available_objects(env)

        # Get description
        scene_description = build_scene_description(available_objects)

        # Compute scene affordances
        affordance_scores = affordance_scoring(possible_actions, available_objects,
                                               block_name="box",
                                               bowl_name="circle",
                                               verbose=False)

        # Get SayCan's plan
        ## Build prompt template
        prompt = base_prompt
        if config_args.rl_script_args.use_env_description:
            prompt += "\n" + scene_description
        prompt += "\n# " + task + "\n"
        actions = []
        for i in range(10):
            ##### Single Lamorel line of code to get scores #####
            raw_llm_scores = lm_server.score([prompt], [possible_actions])[0].tolist()

            ## Compute combined scores given affordance scores
            llm_scores = {_key: _score for _key, _score in zip(possible_actions, raw_llm_scores)}
            combined_scores = {_action: llm_scores[_action] * affordance_scores[_action] for _action in possible_actions}
            combined_scores = normalize_scores(combined_scores)

            ## Select action
            command = max(combined_scores, key=combined_scores.get)
            actions.append(command)
            prompt += command + "\n"

            if config_args.rl_script_args.plot_saycan:
                plot_saycan(llm_scores, affordance_scores, combined_scores, command, show_top=10)

            if command == config_args.rl_script_args.termination_str:
                break

            if config_args.rl_script_args.use_actions_in_env:
                env_action = get_env_action_from_command(command, env, PLACE_TARGETS)
                env.step(env_action)

        print(f"Plan for task {i}:\n" +
              "\n ".join(["{}. {}".format(_idx + 1, _action) for _idx, _action in enumerate(actions)]))

    lm_server.close()

if __name__ == '__main__':
    main()

