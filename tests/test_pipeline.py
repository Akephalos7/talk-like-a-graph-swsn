import unittest
import networkx as nx
from swsn_m2.pipeline import analyze_graph, sample_connected_subgraph, three_encodings, generate_synthetic_graphs

class EDATests(unittest.TestCase):
    def test_directed_degree_and_components(self):
        g=nx.DiGraph([(1,2),(2,1),(2,3)])
        stats=analyze_graph(g,'test')
        self.assertEqual(stats['nodes'],3)
        self.assertEqual(stats['edges'],3)
        self.assertEqual(stats['weak_or_connected_components'],1)
        self.assertEqual(stats['strong_components'],2)
        self.assertAlmostEqual(stats['mean_out_degree'],1.0)
        self.assertAlmostEqual(stats['mean_total_degree'],2.0)
    def test_sample_size_and_directionality(self):
        g=nx.DiGraph([(i,i+1) for i in range(20)])
        h=sample_connected_subgraph(g,12,2026)
        self.assertEqual(h.number_of_nodes(),12)
        self.assertTrue(h.is_directed())
        self.assertTrue(set(h.edges()).issubset(set(g.edges())))
    def test_encoder_semantics(self):
        g=nx.Graph([(1,2),(2,3)])
        txt=three_encodings(g)
        self.assertIn('(1, 2)',txt['edge_list'])
        self.assertIn('Person 1 is friends with person 2',txt['natural_language'])
        d=nx.DiGraph([(3,1)])
        self.assertIn('sent an email',three_encodings(d)['natural_language'])
    def test_self_loops_are_counted_in_eda(self):
        g=nx.DiGraph([(1,1),(1,2),(2,1)])
        stats=analyze_graph(g,'self-loop test')
        self.assertEqual(stats['edges'],3)
        self.assertEqual(stats['self_loops'],1)
    def test_fixed_seed(self):
        x=generate_synthetic_graphs();y=generate_synthetic_graphs()
        self.assertEqual([sorted(g.edges()) for g in x.values()], [sorted(g.edges()) for g in y.values()])

if __name__=='__main__': unittest.main()
