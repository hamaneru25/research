"""Regression: a graph walk must pass the old three-move cutoff and audit its stop."""
import unittest
from fractions import Fraction
from search import Search,make_eta_grid


class Landscape(Search):
    def __init__(self):
        super().__init__(20261020,None,2,eta_points=3)
        self.fake_cache={};self.good={(0,1),(0,2),(0,3),(0,4),(0,5)}

    def seeds(self,m):return [[]]

    def evaluate(self,m,edges,source,force_eta=None):
        edges=tuple(sorted(tuple(sorted(e)) for e in edges))
        if edges not in self.fake_cache:
            value=len(set(edges)&self.good)-.1*len(set(edges)-self.good)
            r=dict(id=str(len(self.records)),edges=edges,deterministic_values=[value]*len(self.etas))
            self.fake_cache[edges]=r;self.records.append(r)
        return self.fake_cache[edges]

    def save(self,status):return None


class ConvergenceTest(unittest.TestCase):
    def test_no_three_move_cutoff_and_complete_terminal_neighborhood(self):
        landscape=Landscape();landscape.local(4,1)
        self.assertEqual(len(landscape.local_runs),3)
        for run in landscape.local_runs:
            self.assertEqual(run['accepted_moves'],5)
            self.assertEqual(run['neighborhood_checks'],6)
            self.assertEqual(len(run['neighbor_ids']),45)
            self.assertLessEqual(run['maximum_neighbor_value'],run['terminal_value']+run['tolerance'])
            self.assertEqual(run['termination_reason'],'no_improving_one_edge_neighbor')

    def test_point_count_is_instance_local_and_all_arrays_follow_it(self):
        small=Search(1,None,2,eta_points=3)
        large=Search(1,None,2,eta_points=20)
        self.assertEqual(small.etas,make_eta_grid(3))
        self.assertEqual(large.etas[0],Fraction(1,20))
        self.assertEqual(large.etas[-1],Fraction(1))
        self.assertEqual(small.weights.shape,(3,63))
        self.assertEqual(large.weights.shape,(20,63))
        for search in [small,large]:
            self.assertEqual(len(search.best),len(search.etas))
            self.assertTrue(all(len(r)==len(search.etas) for r in search.per_m.values()))
            self.assertTrue(all(len(r)==len(search.etas) for r in search.exact_upper.values()))
        with self.assertRaises(ValueError):make_eta_grid(0)


if __name__=='__main__':unittest.main()
