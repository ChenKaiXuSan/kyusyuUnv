# %% 
import argparse
import yaml 
from bisect import bisect
# %%
class State(object):
    def __init__(self, position=(-1, -1), t=0, interval=(0, float('inf'))):
        super().__init__()
        self.position = tuple(position)
        self.time = t 
        self.interval = interval
# %%
class SippGrid(object):
    def __init__(self):
        super().__init__()
        self.interval_list = [(0, float('inf'))]
        self.f = float('inf')
        self.g = float('inf')
        self.parent_state = State()

    def split_interval(self, t, last_t = False):
        """
        function to generate safe-intervals
        """
        for interval in self.interval_list:
            if last_t:
                if t<=interval[0]:
                    self.interval_list.remove(interval)
                elif t>interval[1]:
                    continue
                else:
                    self.interval_list.remove(interval)
                    self.interval_list.append((interval[0], t-1))
            else:
                if t == interval[0]:
                    self.interval_list.remove(interval)
                    if t+1 <= interval[1]:
                        self.interval_list.append((t+1, interval[1]))
                elif t == interval[1]:
                    self.interval_list.remove(interval)
                    if t-1 <= interval[0]:
                        self.interval_list.append((interval[0], t-1))
                elif bisect(interval, t) == 1:
                    self.interval_list.remove(interval)
                    self.interval_list.append((interval[0], t-1))
                    self.interval_list.append((t+1, interval[1]))
            self.interval_list.sort()
        
# %%
class SippGraph(object):
    def __init__(self, map):
        super().__init__()
        self.map = map
        self.dimensions = map["map"]["dimensions"]

        self.obstacles = [tuple(v) for v in map["map"]["obstacles"]]
        self.dyn_obstacles = map["dynamic_obstacles"]

        self.sipp_graph = {}
        self.init_graph()
        self.init_intervals()

    def init_graph(self):
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                grid_dict = {(i,j):SippGrid()}
                self.sipp_graph.update(grid_dict)

    def init_intervals(self):
        if not self.dyn_obstacles: return
        for schedule in self.dyn_obstacles.values():
            # for location in schedule
            for i in range(len(schedule)):
                location = schedule[i]
                last_t = i == len(schedule) - 1 

                position = (location["x"], location["y"])
                t = location["t"]

                self.sipp_graph[position].split_interval(t, last_t)

    def is_valid_position(self, position):
        