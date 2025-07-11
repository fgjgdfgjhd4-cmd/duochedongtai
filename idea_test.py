class NBA:
    def __init__(self, states, alphabet, transition_function, initial_state, accepting_states):
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.initial_state = initial_state
        self.accepting_states = accepting_states

    def accepts(self, input_sequence):
        current_states = {self.initial_state}
        for symbol in input_sequence:
            next_states = set()
            for state in current_states:
                if (state, symbol) in self.transition_function:
                    next_states.update(self.transition_function[(state, symbol)])
            current_states = next_states
            if not current_states:
                return False

        # Check if there is a run that visits an accepting state infinitely often
        while current_states:
            next_states = set()
            for state in current_states:
                if state in self.accepting_states:
                    return True
                for symbol in self.alphabet:
                    if (state, symbol) in self.transition_function:
                        next_states.update(self.transition_function[(state, symbol)])
            current_states = next_states
        return False

# 定义一个简单的非确定性布奇自动机
states = {'q0', 'q1'}
alphabet = {'a', 'b'}
transition_function = {
    ('q0', 'a'): {'q0', 'q1'},
    ('q0', 'b'): {'q0'},
    ('q1', 'a'): {'q1'},
    ('q1', 'b'): {'q1'}
}
initial_state = 'q0'
accepting_states = {'q1'}

nba = NBA(states, alphabet, transition_function, initial_state, accepting_states)

# 测试输入序列
input_sequence = ['a', 'b', 'a', 'a', 'b', 'a']

if nba.accepts(input_sequence):
    print("The input sequence is accepted by the Büchi automaton.")
else:
    print("The input sequence is not accepted by the Büchi automaton.")
