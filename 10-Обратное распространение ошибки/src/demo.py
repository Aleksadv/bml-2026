import math, random

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

class Network:

    def __init__(self, n_in=2, n_hid=2, seed=1):
        rnd = random.Random(seed)
        self.W = [[rnd.uniform(-1, 1) for _ in range(n_in)] for _ in range(n_hid)]
        self.b = [rnd.uniform(-1, 1) for _ in range(n_hid)]
        self.V = [rnd.uniform(-1, 1) for _ in range(n_hid)]
        self.c = rnd.uniform(-1, 1)

    def forward(self, x):

        self.h = [sigmoid(sum(w * xi for w, xi in zip(row, x)) + bj)
                  for row, bj in zip(self.W, self.b)]
        self.o = sigmoid(sum(v * hj for v, hj in zip(self.V, self.h)) + self.c)
        return self.o

    def backward(self, x, y, lr):

        d_o = (self.o - y) * self.o * (1 - self.o)

        d_h = [d_o * v * hj * (1 - hj) for v, hj in zip(self.V, self.h)]

        for j, hj in enumerate(self.h):
            self.V[j] -= lr * d_o * hj
            self.b[j] -= lr * d_h[j]
            for i, xi in enumerate(x):
                self.W[j][i] -= lr * d_h[j] * xi
        self.c -= lr * d_o

    def train(self, data, epochs, lr):
        for epoch in range(epochs):
            loss = 0.0
            for x, y in data:
                o = self.forward(x)
                loss += 0.5 * (y - o) ** 2
                self.backward(x, y, lr)
            if epoch % 2000 == 0:
                print(f"эпоха {epoch:5d}  ошибка {loss:.4f}")

if __name__ == "__main__":
    XOR = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    net = Network()
    net.train(XOR, epochs=10001, lr=0.5)
    for x, y in XOR:
        print(x, "->", round(net.forward(x), 3), "ожидалось", y)
