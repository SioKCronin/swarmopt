import base64
import os
import sys
import tempfile
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from swarmopt.utils.example_html_report import build_example_html_page, png_to_data_uri


SAMPLE_ROWS = [
    {
        'rank': 1,
        'algorithm': 'global',
        'best_cost': '0.001',
        'runtime_s': '1.23',
        'cost_score': 100,
        'time_score': 95,
        'composite_score': 98,
    },
    {
        'rank': 2,
        'algorithm': 'local',
        'best_cost': '0.005',
        'runtime_s': '1.10',
        'cost_score': 80,
        'time_score': 100,
        'composite_score': 88,
    },
]


class TestPngToDataUri(unittest.TestCase):
    def _make_minimal_png(self) -> bytes:
        # Smallest valid 1x1 PNG
        return base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk'
            'YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
        )

    def test_returns_data_uri_prefix(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(self._make_minimal_png())
            path = f.name
        try:
            from pathlib import Path
            uri = png_to_data_uri(Path(path))
            self.assertTrue(uri.startswith('data:image/png;base64,'))
        finally:
            os.unlink(path)

    def test_data_uri_is_valid_base64(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(self._make_minimal_png())
            path = f.name
        try:
            from pathlib import Path
            uri = png_to_data_uri(Path(path))
            b64_part = uri[len('data:image/png;base64,'):]
            decoded = base64.b64decode(b64_part)
            self.assertEqual(decoded, self._make_minimal_png())
        finally:
            os.unlink(path)


class TestBuildExampleHtmlPage(unittest.TestCase):
    def _build(self, rows=None, title='Test Report', intro='<p>Intro</p>',
               image_uri='data:image/png;base64,abc'):
        return build_example_html_page(
            title=title,
            intro_html=intro,
            image_data_uri=image_uri,
            leaderboard_rows=rows if rows is not None else SAMPLE_ROWS,
        )

    def test_returns_string(self):
        self.assertIsInstance(self._build(), str)

    def test_contains_doctype(self):
        self.assertIn('<!DOCTYPE html>', self._build())

    def test_title_appears_in_output(self):
        html = self._build(title='My PSO Report')
        self.assertIn('My PSO Report', html)

    def test_algorithm_names_appear(self):
        html = self._build()
        self.assertIn('global', html)
        self.assertIn('local', html)

    def test_xss_title_escaped(self):
        html = self._build(title='<script>alert(1)</script>')
        self.assertNotIn('<script>alert(1)</script>', html)
        self.assertIn('&lt;script&gt;', html)

    def test_xss_algorithm_escaped(self):
        rows = [{
            'rank': 1,
            'algorithm': '<img src=x onerror=alert(1)>',
            'best_cost': '0.0',
            'runtime_s': '1.0',
            'cost_score': 100,
            'time_score': 100,
            'composite_score': 100,
        }]
        html = self._build(rows=rows)
        self.assertNotIn('<img src=x', html)

    def test_empty_leaderboard(self):
        html = self._build(rows=[])
        self.assertIn('<tbody>', html)
        self.assertNotIn('<tr>', html.split('<tbody>')[1].split('</tbody>')[0])

    def test_image_data_uri_embedded(self):
        uri = 'data:image/png;base64,FAKEDATA'
        html = self._build(image_uri=uri)
        self.assertIn(uri, html)

    def test_composite_score_wrapped_in_strong(self):
        html = self._build()
        self.assertIn('<strong>98</strong>', html)
        self.assertIn('<strong>88</strong>', html)


if __name__ == '__main__':
    unittest.main()
