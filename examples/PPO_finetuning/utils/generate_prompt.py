def generate_prompt(obs, infos):
    prompt = "{}\n".format(infos["goal"])
    prompt += "Observation: {}\n".format(', '.join(obs))
    prompt += "Action:"
    return prompt