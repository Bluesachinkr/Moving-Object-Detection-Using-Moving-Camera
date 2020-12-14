"""
Created on Fri Oct 25 22:05:15 2020
@author: Sachin Kumar
"""
import sys
import math
import numpy as np

np.seterr(over='ignore')

# Partition region matching class
class PartitionRegionMatching:
    class MotionVector:

        def __init__(self):
            self.region = -1
            self.minSum = 0
            self.minX = -1
            self.miny = -1
            self.minVector = -1

    def __init__(self, h, w, previousGraph, currentFrame):
        self.h = h
        self.w = w
        self.previousGraph = previousGraph
        self.previousFrame = self.previousGraph.colorFrame
        self.currentFrame = currentFrame
        self.motionEstimation()

    def updateGraph(self):
        return self.previousGraph

    def motionEstimation(self):
        graphNode = self.previousGraph.graphNode

        for index in range(0, self.previousGraph.totalRegion):
            node = graphNode[index]
            topx = node.topX
            topy = node.topY
            bottonx = node.bottomX
            bottomy = node.bottomY
            area = node.area

            if area > 64:
                mvx, mvy, probability = self.motionEstimationUtil(node)
                if probability > 0: node.mvx, node.mvy = mvx, mvy

            else:
                vk, vkx, vky, vp = self.vectorPoints(topx, topy, bottonx + 1, bottomy + 1)
                if vk > 0: node.mvx, node.mvy = vkx, vky

            node.pr = self.motionReliability(node.mvx, node.mvy, topx, topy, bottonx + 1,
                                             bottomy + 1)
            arr = []
            arr.append(index)
            arr.append(node.mvx)
            arr.append(node.mvy)
            arr.append(node.pr)
            print(arr)
            graphNode[index] = node

        self.previousGraph.graphNode = graphNode

    def motionEstimationUtil(self, node):
        length = node.length
        height = node.height

        h, l, eh, el, n = self.splitRegion(height, length)
        row = node.topX
        col = node.topY
        movingRegion = []

        for cnt in range(1, n):
            self.motionVectorUtil(row, col, h, l, movingRegion)
            row += h
            col += l
        self.motionVectorUtil(row, col, eh, el, movingRegion)

        return self.probabilityFunction(movingRegion)

    def motionVectorUtil(self, row, col, h, l, movingRegion):
        vk, vkx, vky, minvp = self.vectorPoints(row, col, row + h, col + l)
        obj = self.MotionVector()
        obj.minX = vkx
        obj.miny = vky
        obj.minSum = vk
        obj.minVector = minvp
        movingRegion.append(obj)

    def vectorPoints(self, x1, y1, x2, y2):
        vk, vkx, vky, vp, cnt = sys.maxsize, -1, -1, -1, 1
        for x in range(-4, 5):
            for y in range(-4, 5):
                cnt += 1
                sum = self.sumAbsoluteDifference(x, y, x1, y1, x2, y2)
                if vk > sum: vk, vkx, vky, vp = sum, x, y, cnt

        return vk, vkx, vky, vp

    def probabilityFunction(self, movingRegion):
        # probability function
        cnt = 1
        mvx = 0
        mvy = 0
        maxprob = 0
        for i in range(-4, 5):
            for j in range(-4, 5):
                cnt += 1
                sum = 0
                for region in movingRegion:
                    if cnt == region.minVector:
                        sum += 1
                sum /= 81
                if sum > maxprob:
                    mvx, mvy = i, j
                    maxprob = sum

        return mvx, mvy, maxprob

    def sumAbsoluteDifference(self, vx, vy, x1, y1, x2, y2):
        sum = 0
        for x0 in range(x1, x2):
            for y0 in range(y1, y2):
                s = x0 + vx
                t = y0 + vy
                if (s < 0 or t < 0 or s >= self.h or t >= self.w): continue
                for z0 in range(0, 3):
                    sum += math.fabs(self.previousFrame[x0, y0, z0] - self.currentFrame[s, t, z0])
        return sum

    def min(self, a, b):
        if a > b:
            return b
        else:
            return a

    def splitRegion(self, height, length):
        a = int(math.sqrt(height))
        b = int(math.sqrt(length))
        subregion = self.min(a, b)

        minlength = int(length / subregion)
        minheight = int(height / subregion)

        extralength = length - (minlength * subregion)
        extraheight = height - (minheight * subregion)

        totalRegion = subregion + 1

        return minheight, minlength, extraheight, extralength, totalRegion

    def motionReliability(self, mvx, mvy, x1, y1, x2, y2):
        numerator = 0
        denominator = self.sumAbsoluteDifference(mvx, mvy, x1, y1, x2, y2) + 1
        for x in range(x1, x2):
            for y in range(y1, y2):
                m = 0
                for i in range(0, 3):
                    temp = math.fabs(self.derivative(1, x, y, i)) + math.fabs(
                        self.derivative(0, x, y, i))
                    m += temp
                numerator += (m / denominator)

        return numerator / denominator

    def derivative(self, wrt, x, y, z):
        if x == 0 or y == 0:
            return self.currentFrame[x, y, z]
        else:
            if wrt == 1:
                return self.currentFrame[x, y, z] - self.currentFrame[x - 1, y, z]
            else:
                return self.currentFrame[x, y, z] - self.currentFrame[x, y - 1, z]
