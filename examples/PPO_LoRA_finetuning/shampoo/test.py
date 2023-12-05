from pettingzoo.test import api_test, render_test
from shampoo import shampoo_v0

if __name__ == "__main__":
    # api test
    env = shampoo_v0.env()
    api_test(env)

    # render test
    print("\nStarting render test")
    env_func = shampoo_v0.env
    render_test(env_func)
    print("Passed render test")