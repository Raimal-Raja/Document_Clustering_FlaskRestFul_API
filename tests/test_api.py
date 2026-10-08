import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


class ClusteringApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ['CLUSTERING_DB_PATH'] = str(Path(self.temp.name) / 'test.db')
        path = Path(__file__).resolve().parents[1] / 'Task_2' / 'app.py'
        spec = importlib.util.spec_from_file_location('clustering_test_app', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.client = self.module.app.test_client()

    def tearDown(self):
        os.environ.pop('CLUSTERING_DB_PATH', None)
        self.temp.cleanup()

    def test_initialization_preserves_custom_documents(self):
        with self.module.get_db() as conn:
            conn.execute('INSERT INTO Documents VALUES (99, ?, ?)', ('Custom', 'Keep me'))
        self.module.init_db()
        self.assertEqual(self.client.get('/api/stats').json['total_docs'], 21)

    def test_assignment_checks_json_and_foreign_keys(self):
        self.assertEqual(self.client.post('/api/assign', json=[]).status_code, 400)
        self.assertEqual(self.client.post('/api/assign', json={'doc_id': 999, 'cluster_id': 1}).status_code, 400)
        self.assertEqual(self.client.post('/api/assign', json={'doc_id': 1, 'cluster_id': 2}).status_code, 200)

if __name__ == '__main__':
    unittest.main()
