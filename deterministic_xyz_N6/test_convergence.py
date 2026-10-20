"""Regression: a graph walk must pass the old three-move cutoff and audit its stop."""
import unittest
from search import Search


class Landscape(Search):
    def __init__(self):
        super().__init__(20261020,None,2)
        self.fake_cache={};self.good={(0,1),(0,2),(0,3),(0,4),(0,5)}

    def seeds(self,m):return [[]]

    def evaluate(self,m,edges,source,force_eta=None):
        edges=tuple(sorted(tuple(sorted(e)) for e in edges))
        if edges not in self.fake_cache:
            value=len(set(edges)&self.good)-.1*len(set(edges)-self.good)
            r=dict(id=str(len(self.records)),edges=edges,deterministic_values=[value]*10)
            self.fake_cache[edges]=r;self.records.append(r)
        return self.fake_cache[edges]

    def save(self,status):return None


class ConvergenceTest(unittest.TestCase):
    def test_no_three_move_cutoff_and_complete_terminal_neighborhood(self):
        landscape=Landscape();landscape.local(4,1)
        self.assertEqual(len(landscape.local_runs),7)
        for run in landscape.local_runs:
            self.assertEqual(run['accepted_moves'],5)
            self.assertEqual(run['neighborhood_checks'],6)
            self.assertEqual(len(run['neighbor_ids']),45)
            self.assertLessEqual(run['maximum_neighbor_value'],run['terminal_value']+run['tolerance'])
            self.assertEqual(run['termination_reason'],'no_improving_one_edge_neighbor')


if __name__=='__main__':unittest.main()
