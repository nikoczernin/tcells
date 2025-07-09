# Environment base class for reinforcement learning tasks.
# Structure for state/action handling, transition dynamics,
# reward computation, and simulation utilities for both discrete domains
# Note to self: look into continuous states

from pprint import pprint
import random

class Environment():
    def __init__(self, actions):
        # Initialize environment with action set
        self.actions = actions
        self.rewards = {a: 0 for a in self.actions}
        self.starting_state = None
        self.terminal_states = []

    def set_start(self, starting_state=None):
        # Set the initial state of the environment
        # input: starting_state (optional) - any valid state
        # output: none
        if starting_state is not None:
            self.starting_state = starting_state

    def reset(self):
        pass

    def state_generator(self):
        # Generator for all possible states
        # input/output: none (to be overridden by subclass)
        # yield ...
        raise NotImplementedError()

    def terminal_state_generator(self):
        # Yields only terminal states from the full state set
        for state in self.state_generator():
            if self.state_is_terminal(state):
                # yield ...
                raise NotImplementedError()

    def state_is_terminal(self, state) -> bool:
        # Checks if a state is terminal
        # input: state - any state
        # output: bool
        raise NotImplementedError()

    @staticmethod
    def get_random_state(states, states_to_avoid=None):
        # Returns a random state, avoiding specific ones if needed
        # input: states - list, states_to_avoid - list or None
        # output: single state
        s = random.choice(states)
        if states_to_avoid is not None:
            if s in states_to_avoid:
                s = Environment.get_random_state(states, states_to_avoid)
        return s

    def is_this_action_possible(self, state, action) -> bool:
        # Checks if action is in the environment's action list
        # input: state - any, action - any
        # output: bool
        return True if action in self.actions else False

    def apply_action(self, state, action):
        # Computes next state from applying action in current state
        # input: state, action
        # output: new_state (to be overridden)
        raise NotImplementedError()

    def get_reward(self, state: tuple, action: tuple, new_state: tuple = None):
        raise NotImplementedError()
