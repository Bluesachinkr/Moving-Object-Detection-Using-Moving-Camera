import math
import sys
import random
from collections import defaultdict


class MRF:
    def __init__(self, graph):
        self.graph = graph
        self.h, self.w, _ = self.graph.frame.shape
        self.curlabel = 1
        self.stableNodes = 0
        self.labels = [0] * (self.graph.totalRegion)
        self.visited = [False] * (self.graph.totalRegion)
        self.energy = [sys.maxsize] * (self.graph.totalRegion)

    def graphOptmization(self):
        totalNodes = self.graph.totalRegion
        maxRandomNumber = self.graph.totalRegion - 1

        while self.stableNodes != totalNodes:
            index = random.randint(1, maxRandomNumber)
            self.energyFunction(index)

        map = defaultdict(set)
        for index in range(1, self.graph.totalRegion):
            label = self.labels[index]
            map[label].add(index)

        region = 1
        regionMap = defaultdict(set)
        for setvalue in map.values():
            regionMap[region] = setvalue
            region += 1
        self.graph.labels = self.labels

        return regionMap, region,self.graph

    def energyFunction(self, index):

        adjacentRegion = self.graph.adjacentRegion[index]

        u1 = self.U1(index, adjacentRegion)
        u2 = self.U2(index, adjacentRegion)
        for region in self.graph.adjacentRegion[index]:
            if self.labels[index] == self.labels[region]:
                u1 += 1

        u = u1 + u2
        if self.energy[index] > u:
            self.energy[index] = u
            self.stableNodes += 1
            self.curlabel += 1
            self.labels[index] = self.curlabel
            for region in self.graph.adjacentRegion[index]:
                if self.visited[region] == True:
                    self.stableNodes -= 1
                    self.visited[region] = False
                self.labels[region] = self.labels[index]

    def U1(self, index, regions):
        mvx = self.graph.graphNode[index].mvx
        mvy = self.graph.graphNode[index].mvy
        pr = self.graph.graphNode[index].pr
        sum = 0
        for r in regions:
            if self.labels[index] == self.labels[r]:
                mvx1 = self.graph.graphNode[r].mvx
                mvy1 = self.graph.graphNode[r].mvy
                pr1 = self.graph.graphNode[r].pr
                v = self.euclideon(mvx, mvy, mvx1, mvy1) * math.sqrt(pr * pr1)
                sum += v
        return sum

    def U2(self, index, regions):
        node = self.graph.graphNode[index]
        gxa = node.topX + ((node.bottomX - node.topX) / 2)
        gya = node.topY + ((node.bottomY - node.topY) / 2)
        sum = 0
        for r in regions:
            if self.labels[index] == self.labels[r]:
                nodetemp = self.graph.graphNode[r]
                gxb = nodetemp.topX + ((nodetemp.bottomX - nodetemp.topX) / 2)
                gyb = nodetemp.topY + ((nodetemp.bottomY - nodetemp.topY) / 2)
                a = gxa - gxb
                a = a * a
                b = gya - gyb
                b = b * b
                v = 1 / (1 + math.sqrt((a + b)))
                sum -= v

        return sum

    def euclideon(self, mvx, mvy, mvx1, mvy1):
        a = mvx - mvx1
        b = mvy - mvy1
        a = a * a
        b = b * b
        distance = a + b
        return math.sqrt(distance)

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

    def limit(self, x, y):
        return 0 <= x < self.h and 0 <= y < self.w
