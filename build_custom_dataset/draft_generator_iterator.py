
def my_generator():
    for i in range(10):
        yield i

gen = my_generator()
iter = iter(gen)


print(type(gen))
print(type(iter))

for value in gen:
    print(value)













