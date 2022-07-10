import json
import os
import numpy as np

# you can clear the dict in a dict separately
a = {'a': [1, 2, 3, 4, 5], 'b': {'1': 1, '2': 2 }}
a['b'].clear()
print(a)

# python format set digits
blah = 999543
print(f"b{blah:06}")

# json dumping 
a = {
    'hello': 1, 
    'my': {
        'bruh': 1002, 
        'cool': 9999
    }
}

# w = open(os.path.join(os.getcwd(), 'haha.json'), 'w') 
# json.dump(a, w)

# list stuff
a = [1]
if a[0] in [1, 2, 3, 4]:
    print("a in list")

# enumeration
print(enumerate([1, 2, 3, 4]))

# masking stuff
rand = np.random.rand(10, 4)
agents = [[1], [1], [1], [1], [2], [2], [2], [3], [3], [4]]
rand = np.concatenate((rand, agents), axis=1)
agent_ids = [1, 2, 3, 4]
masks = np.array([(rand[:, 4] == agent_id) for agent_id in agent_ids])
print(masks)

a = np.empty((0, 5))
a = np.append(a, [1, 2, 3, 4, 5, 6])
print(a)