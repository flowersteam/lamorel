def generate_prompt(obs):
    prompt = "Goal of the agent: {}\n".format(obs["mission"])
    prompt += "Observation: {}\n".format(', '.join(obs['descriptions']))
    prompt += "Action: "
    return prompt