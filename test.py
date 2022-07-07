from abc import ABC, abstractmethod


class A(ABC):
    def __init__(self, *args, **kwargs):
        self.a = 'A'

    @abstractmethod
    def reset(self):
        print('a reset')
        self.a = 0

class B():
    def __init__(self, b):
        self.b = b

    def reset(self):
        print('b reset')
        self.b = 0

    def only_b(self):
        print('only b')


class C(A, B):
    def __init__(self, a, b, c):
        A.__init__(self)
        B.__init__(self, b)
        self.c = c 
        print(self.a, self.b, self.c)
        self.reset()
        print(self.a, self.b, self.c)

    def reset(self):
        print('starting c reset')
        B.reset(self)
        A.reset(self)

if __name__ == '__main__':
    o = C(a=1, b=2, c=3)
    print(o.a, o.b, o.c)
    o.only_b()