def generate_prompt(obs, infos):
    prompt = "Goal of the agent: {}\n".format(obs["mission"])
    prompt += "Observation: {}\n".format(infos['descriptions'])
    prompt += "Action: "
    return prompt