

it = iter(range(10))

print(type(it))
#############################
r = range(10)

for _ in range(3):
    print(next(it))

#############################
print(list(r))

print(r)
for _ in r:
    print(_)

#############################
 