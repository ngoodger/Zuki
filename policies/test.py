class Parent():
    def __init__(self, test0, test1):
        print(test0)
        print(test1)


class Child(Parent):
    def __init__(self, test2, test3, *args, **kargs):
        super().__init__(*args, **kargs)
        print(test2)
        print(test3)


Child("Is", "Niko", "My", test0="Name")
