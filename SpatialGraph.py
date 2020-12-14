import numpy as np
from collections import defaultdict

class SpatialGraph():

    def __init__(self, totalRegion, regionFrame, colorFrame, frame):
        self.totalRegion = totalRegion
        self.adjacentRegion = defaultdict(set)
        self.regionFrame = regionFrame
        self.h, self.w = self.regionFrame.shape
        self.frame = frame
        self.colorFrame = colorFrame
        self.graphNode = self.getNodes()

        self.labels = None

        self.createSpatialGraph()

    def addEdge(self, src, dest):
        self.adjacentRegion[src].add(dest)
        self.adjacentRegion[dest].add(src)

    class AdjNode():
        def __init__(self, region):
            self.region = region
            self.mvx = -4
            self.mvy = -4
            self.pr = 0
            self.area, self.length, self.height = 0, 0, 0
            self.topX, self.topY, self.bottomX, self.bottomY = -1, -1, -1, -1

        def setArea(self, length, height, area):
            self.length = length
            self.height = height
            self.area = area

        def setPoints(self, topX, topY, bottomX, bottomY):
            self.topX = topX
            self.topY = topY
            self.bottomX = bottomX
            self.bottomY = bottomY

    def getNodes(self):
        hmin, hmax, vmin, vmax = self.coordinatesBox()
        node = [None] * (self.totalRegion)

        for index in range(0, self.totalRegion):
            x = vmin[index]
            y = hmin[index]
            x1 = vmax[index]
            y1 = hmax[index]
            height = x1 - x + 1
            length = y1 - y + 1
            area = length * height
            tempNode = self.AdjNode(index)
            tempNode.setArea(length, height, area)
            tempNode.setPoints(x, y, x1, y1)
            node[index] = tempNode
        return node

    def coordinatesBox(self):
        hmin = {}
        hmax = {}
        vmin = {}
        vmax = {}
        for i in range(0, self.h):
            for j in range(0, self.w):
                value = self.regionFrame[i, j]
                if (value in vmin.keys()) == False:
                    vmin[value] = i
                    vmax[value] = i
                else:
                    if i > vmax[value]:
                        vmax[value] = i

                    if i < vmin[value]:
                        vmin[value] = i

                if (value in hmin.keys()) == False:
                    hmin[value] = j
                    hmax[value] = j
                else:
                    if j > hmax[value]:
                        hmax[value] = j

                    if j < hmin[value]:
                        hmin[value] = j

        return hmin, hmax, vmin, vmax

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

    def adjacentRegions(self):
        graph = defaultdict(set)
        visited = np.full((self.h, self.w), False, dtype=bool)

        queue = []
        queue.append((0, 0))

        while queue:
            a, b = queue.pop(0)
            value = self.regionFrame[a, b]
            if visited[a, b] == False:
                neighbours = self.getNeighbour(a, b)
                for x, y in neighbours:
                    k = (int)(self.regionFrame[x, y])
                    if visited[x, y] == False:
                        queue.append((x, y))
                    if value != k:
                        graph[value].add(k)

            visited[a, b] = True

        return graph

    def createSpatialGraph(self):
        adjacentRegion = self.adjacentRegions()

        for num in range(1, self.totalRegion):
            for region in adjacentRegion[num]:
                self.addEdge(num, region)
