if input == "":

    import collections
    import math
    import random

    gamma = random.gammavariate
    sqrt = math.sqrt
    log = math.log
    R, P, S = 0, 1, 2
    index = {"R": R, "P": P, "S": S}
    beat = (P, S, R)
    name = ("R", "P", "S")

    def ucb(s, n, t):
        return s + sqrt(2 * log(t) / n)

    def belief(xs):
        n = sum(xs)
        m = sum(1 for x in xs if x)
        if m == 0:
            a = 1.0
        else:
            a = m / (6.0 * log((n + 1.0) / m))
        return [gamma(x + a, 1) for x in xs]

    class MarkovTree:
        def __init__(self, parent = None):
            self.counts = [0.0 for _ in range(3)]
            self.visits = [0.0 for _ in range(3)]
            self.children = None
            self.parent = parent

        def score(self, t):
            return ucb(self.pos_total, self.neg_total, self.total_visits, t)

        def select_move(self, t):
            r, p, s = belief(self.counts)
            visits = belief(self.visits)
            scores = [s - p, r - s, p - r]
            scores = [ucb(s, k, t) for s, k in zip(scores, visits)]
            best = max(scores)
            return (best, scores.index(best))

        def update(self, h, i):
            for n in h:
                self.counts[i] += 1
                if self.children is None:
                    break
                self = self.children[n]
                if self is None:
                    break

        def predict(self, h):
            stop = False
            t = 1
            for d, n in enumerate(h):
                t += sum(self.visits)
                if stop or d >= 16:
                    break
                if self.children is None:
                    self.children = [None for _ in range(3)]
                if self.children[n] is None:
                    self.children[n] = MarkovTree(self)
                    stop = True
                child = self.children[n]
                self = child
            best_score = float("-inf")
            leaf = self
            while self is not None:
                score, move = self.select_move(t)
                if score >= best_score:
                    best_score = score
                    best_move = move
                self = self.parent
            return (leaf, best_move)

    tree = MarkovTree()
    history = collections.deque([])

    node = tree

else:

    i = index[input]
    j = index[output]

    while node is not None:
        node.visits[j] += 1
        node = node.parent
    tree.update(history, i)

    history.appendleft(i)
    history.appendleft(j)

node, k = tree.predict(history)
output = name[k]