import json
import os

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
