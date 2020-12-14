import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class RegionGrow():
    def __init__(self, h, w, alpha, beta):
        self.h, self.w = h, w
        self.alpha = alpha
        self.beta = beta
        self.frequency = defaultdict(int)
        self.smallregion = []

        self.im = None
        self.passedBy = np.zeros((self.h, self.w), np.long)
        self.currentRegion = 1
        self.queue = []
        self.SEGS = np.zeros((self.h, self.w, 3), dtype='uint8')

    def getNeighbour(self, x0, y0):
        neighbour = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if (i, j) == (0, 0):
                    continue
                x = x0 + i
                y = y0 + j
                if self.limit(x, y):
                    neighbour.append((x, y))
        return neighbour

    def applyRegionGrow(self, frame):
        self.im = frame
        self.grown(int(self.h / 2), int(self.w / 2))
        self.splitRegion()
        self.regionSize()
        graph = self.adjacentRegions()
        result = defaultdict(int)
        for region in self.smallregion:
            adjacentRegion = graph[region]
            maxarea = 0
            index = 0
            for adj in adjacentRegion:
                a = self.frequency[adj]
                if a > maxarea:
                    maxarea = a
                    index = adj
            result[region] = index
        hashset = set()
        for i in range(0, self.h):
            for j in range(0, self.w):
                val = self.passedBy[i, j]
                if val in result.keys():
                    self.passedBy[i, j] = result[val]
                hashset.add(self.passedBy[i, j])
        result = defaultdict(int)
        index = 0
        for adj in hashset:
            result[adj] = index
            index += 1
        for i in range(0, self.h):
            for j in range(0, self.w):
                val = self.passedBy[i, j]
                self.passedBy[i, j] = result[val]
        for i in range(0, self.h):
            for j in range(0, self.w):
                val = self.passedBy[i][j]
                self.SEGS[i][j] = val * 35, val * 90, val * 30
        return self.SEGS, self.passedBy, index

    def ApplyRegionGrow(self, frame):
        self.im = frame
        self.grown(int(self.h / 2), int(self.w / 2))
        self.printFrame()
        for i in range(0, self.h):
            for j in range(0, self.w):
                self.grown(i, j)
        self.printFrame()
        # optmization of regions
        while self.regionSize() > 50:
            adjacentRegion = self.adjacentRegions()
            self.optimizedRegions(adjacentRegion)
            # self.printFrame()

        return self.SEGS, self.passedBy, self.currentRegion, self.smallregion

    def printFrame(self):
        plt.imshow(self.SEGS)
        plt.show()

    def grown(self, x0, y0):
        if self.passedBy[x0, y0] == 0 and (
                int(self.im[x0, y0, 0]) * int(self.im[x0, y0, 1]) * int(self.im[x0, y0, 2]) > 0):
            self.passedBy[x0, y0] = self.currentRegion
            self.frequency[self.currentRegion] = 1
            self.queue.append((x0, y0))
            while self.queue:
                x, y = self.queue.pop(0)
                self.BFS(x, y)

            self.currentRegion += 1

    def BFS(self, x0, y0):
        regionNum = self.passedBy[x0, y0]
        elems = []
        elems.append((int(self.im[x0, y0, 0]) + int(self.im[x0, y0, 1]) + int(self.im[x0, y0, 2])) / 3)
        var = self.alpha
        neighbours = self.getNeighbour(x0, y0)

        for x, y in neighbours:
            if self.passedBy[x, y] == 0 and self.distance(x, y, x0, y0) < var:
                self.passedBy[x, y] = regionNum
                self.frequency[self.currentRegion] += 1
                self.queue.append((x, y))
                elems.append((int(self.im[x, y, 0]) + int(self.im[x, y, 1]) + int(self.im[x, y, 2])) / 3)
                var = np.var(elems)
            var = max(var, self.alpha)

    def limit(self, x, y):
        return 0 <= x < self.h and 0 <= y < self.w

    def distance(self, x, y, x0, y0):
        return ((int(self.im[x, y, 0]) - int(self.im[x0, y0, 0])) ** 2 + (
                int(self.im[x, y, 1]) - int(self.im[x0, y0, 1])) ** 2 + (
                        int(self.im[x, y, 2]) - int(self.im[x0, y0, 2])) ** 2) ** 0.5

    def clear(self):
        self.passedBy = np.zeros((self.h, self.w), np.double)
        self.currentRegion = 1
        self.queue = []
        self.SEGS = np.zeros((self.h, self.w, 3), dtype='uint8')

    def splitRegion(self):
        self.currentRegion = 1
        for i in range(0, self.h):
            for j in range(0, self.w):
                if self.passedBy[i, j] == 0: self.splitRegionUtil(i, j)

    def splitRegionUtil(self, x, y):
        queue = []
        queue.append((x, y))
        while queue:
            x, y = queue.pop(0)
            if self.passedBy[x, y] == 0:
                neighbour = self.getNeighbour(x, y)
                for a, b in neighbour:
                    if self.passedBy[a, b] == 0: queue.append((a, b))
                self.passedBy[x, y] = self.currentRegion
        self.currentRegion += 1

    def regionSize(self):
        self.smallregion = set()
        for i in range(0, self.h):
            for j in range(0, self.w):
                val = self.passedBy[i, j]
                if val in self.frequency.keys():
                    self.frequency[val] += 1
                else:
                    self.frequency[val] = 1
        for index in range(0, self.currentRegion):
            if self.frequency[index] <= self.beta: self.smallregion.add(index)

    def adjacentRegions(self):
        graph = defaultdict(set)
        visited = np.full((self.h, self.w), False, dtype=bool)
        queue = []
        queue.append((0, 0))
        while queue:
            a, b = queue.pop(0)
            value = self.passedBy[a, b]
            if visited[a, b] == False:
                neighbours = self.getNeighbour(a, b)
                for x, y in neighbours:
                    k = self.passedBy[x, y]
                    if visited[x, y] == False:
                        queue.append((x, y))
                    if value != k:
                        graph[value].add(k)
            visited[a, b] = True
        return graph

    def optimizedRegions(self, graph):
        visited = [True] * (self.currentRegion)
        mergingRegion = defaultdict(set)
        for region in self.smallregion:
            area = self.frequency[region]
            if area > self.beta: continue
            adjacentregion = graph[region]
            minarea = (self.h * self.w) + 1
            index = 0
            for adjacent in adjacentregion:
                if visited[adjacent] == False: continue
                curarea = self.frequency[adjacent]
                if curarea < minarea:
                    minarea = curarea
                    index = adjacent

            visited[region] = False
            self.frequency[index] = area + minarea
            mergingRegion[index].add(region)
            for adj in mergingRegion[region]: mergingRegion[index].add(adj)
            for adj in adjacentregion: graph[index].add(adj)

        size = self.currentRegion
        self.currentRegion = 1
        result = defaultdict(int)
        for k in range(1, size):
            if visited[k] == False:  continue
            result[k] = self.currentRegion
            area = 1
            for adj in mergingRegion[k]:
                result[adj] = self.currentRegion
                area += 1
            self.frequency[self.currentRegion] = area
            self.currentRegion += 1

        for i in range(0, self.h):
            for j in range(0, self.w):
                v = result[self.passedBy[i, j]]
                self.passedBy[i, j] = v
