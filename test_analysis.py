import math
import unittest
from unittest.mock import patch

import numpy as np
from analysis import aggregate, analyze, clip_points, curve, event_region, fit_curve, mt_outliers, trajectory_metrics, trial_range


class AnalysisTests(unittest.TestCase):
    def test_width_uses_perpendicular_axis(self):
        direction = np.array([math.sqrt(3) / 2, -.5])
        normal = np.array([.5, math.sqrt(3) / 2])
        coords = np.outer([0, 10, 20], direction) + np.outer([-1, 0, 1], normal)
        points = np.column_stack(([0, 500, 1000], coords))
        expected = 2 * math.sqrt(101) / 4.133
        self.assertAlmostEqual(trajectory_metrics(points, 1000), expected)
        shifted = points.copy()
        shifted[:, 1:] += 100
        self.assertAlmostEqual(trajectory_metrics(shifted, 1000), expected)

    def test_clip_interpolates_both_boundaries(self):
        points = np.array([[0, 0, 0], [10, 20, 10], [20, 40, 0]])
        np.testing.assert_allclose(clip_points(points, 5, 15), [[5, 10, 5], [10, 20, 10], [15, 30, 5]])
        self.assertIsNone(clip_points(points, None, 15))
        self.assertIsNone(clip_points(points, 0, 21))

    def test_event_overlap(self):
        self.assertEqual(event_region([dict(startMs=1, endMs=3)], 5, 10), 'outside')
        self.assertEqual(event_region([dict(startMs=4, endMs=7)], 5, 10), 'both')
        self.assertEqual(event_region([dict(startMs=6, endMs=7)], 5, 10), 'inside')
        self.assertEqual(event_region([dict(startMs=5, endMs=5)], 5, 10), 'inside')
        self.assertEqual(event_region([dict(startMs=1, endMs=3)], None, 10), 'unknown')

    def test_equal_participants_and_er_denominator(self):
        rows = []
        for p, n, success, mt, er in [('001', 1, True, 100, 1), ('001', 2, True, 300, 0),
                                     ('002', 1, True, 1000, 0), ('002', 2, False, 5000, 0)]:
            rows.append(dict(participant=p, n=n, success=success, phase='post', condition='C1',
                             source='test.csv', line=n, attempt=n, errorType='deviation' if er else 'none',
                             region='inside' if er else 'none', coreMtMs=dict(MT=mt, ER=er, TPe=1)))
        options = dict(scope='coreMtMs', phase='post', mode='mean', bin=10,
                       participants=['001', '002'], conditions=['C1'], exclude=True)
        result = analyze(rows, options)
        self.assertEqual(result['included'], 3)
        self.assertEqual(result['summary'][0]['MT'], 600)
        self.assertEqual(result['summary'][0]['ER'], 25)
        self.assertEqual(len(result['excluded']), 1)
        self.assertEqual(result['series'][0]['x'], [5.5])
        self.assertEqual(aggregate(rows[:2], 'ER', 'coreMtMs'), 50)

    def test_missing_er_is_not_zero(self):
        self.assertIsNone(aggregate([{'coreMtMs': {'ER': None}}], 'ER', 'coreMtMs'))
        self.assertIsNone(trajectory_metrics(np.array([[0, 0, 0], [1, 0, 0]]), 1))

    def test_regression_recovers_curve(self):
        x = np.arange(1, 101, dtype=float)
        y = curve(x, [300, 1500, 4, .6])
        result = fit_curve(x, y)
        self.assertTrue(result['ok'])
        self.assertLess(result['rmse'], .01)
        self.assertGreater(result['r2'], .99999)
        self.assertFalse(fit_curve([1, 2], [5, 4])['ok'])

    def test_inverse_curve_units(self):
        x = np.arange(1, 101, dtype=float)
        y = curve(x, [.01, .05, 4, .6])
        result = fit_curve(x, y, unit_scale=1000)
        self.assertTrue(result['ok'])
        self.assertLess(result['rmse'], 1e-7)
        self.assertGreater(result['r2'], .99999)
        p = result['parameters']
        np.testing.assert_allclose(curve(x, [p[k] for k in ('A', 'B', 'E', 'a')]), y, atol=1e-7)
        np.testing.assert_allclose(result['y'], curve(result['x'], [.01, .05, 4, .6]), atol=1e-7)

    def test_inverse_before_averaging(self):
        rows = []
        for p, tpe in [('001', 2), ('001', 4), ('002', 10), ('002', None), ('002', 0)]:
            rows.append(dict(participant=p, n=1, success=True, phase='practice', condition='C1',
                             errorType='none', region='none', coreMtMs=dict(MT=100, ER=0, TPe=tpe),
                             mtMs=dict(MT=200, ER=0, TPe=tpe * 2 if tpe else tpe)))
        options = dict(scope='coreMtMs', phase='practice', mode='mean', bin=10,
                       participants=['001', '002'], conditions=['C1'], exclude=True)
        with patch('analysis.fit_curve', return_value={'ok': False, 'message': 'test'}):
            result = analyze(rows, options)
            self.assertAlmostEqual(result['inverseFits'][0]['observedY'][0], ((.5 + .25) / 2 + .1) / 2)
            options['bin'] = 100
            self.assertEqual(analyze(rows, options)['inverseFits'], result['inverseFits'])
            options['scope'] = 'mtMs'
            self.assertAlmostEqual(analyze(rows, options)['inverseFits'][0]['observedY'][0], .11875)
            options['phase'] = 'post'
            self.assertEqual(analyze(rows, options)['inverseFits'], [])

    def test_inverse_selection_and_exclusion(self):
        rows = [dict(participant='001', n=i + 1, success=i != 1, phase='practice', condition='C1',
                     source='test.csv', line=i+2, attempt=i+1, errorType='none', region='none',
                     coreMtMs=dict(MT=100, ER=0, TPe=value)) for i, value in enumerate([2, 4, None])]
        options = dict(scope='coreMtMs', phase='practice', mode='individual', bin=10,
                       participants=['001'], conditions=['C1'], exclude=True)
        result = analyze(rows, options)
        self.assertEqual(result['inverseFits'][0]['observedX'], [1])
        self.assertEqual(result['inverseFits'][0]['observedY'], [.5])
        self.assertFalse(result['inverseFits'][0]['ok'])
        options['exclude'] = False
        self.assertEqual(analyze(rows, options)['inverseFits'][0]['observedY'], [.5, .25])
        options['participants'] = []
        self.assertEqual(analyze(rows, options)['inverseFits'], [])

    def test_trial_range_filters_all_outputs(self):
        rows = [dict(participant='001', n=n, success=n not in (2, 12), phase='practice', condition='C1',
                     source='test.csv', line=n+1, attempt=n, errorType='deviation', region='inside',
                     coreMtMs=dict(MT=n*100, ER=1, TPe=n)) for n in range(1, 21)]
        options = dict(scope='coreMtMs', phase='practice', mode='mean', bin=10,
                       participants=['001'], conditions=['C1'], exclude=True, trialStart=11, trialEnd=20)
        result = analyze(rows, options)
        self.assertEqual(result['selected'], 10)
        self.assertEqual(result['included'], 9)
        self.assertEqual([r['n'] for r in result['excluded']], [12])
        self.assertEqual(result['regions']['inside'], 9)
        self.assertAlmostEqual(result['summary'][0]['MT'], (sum(range(11,21))-12)*100/9)
        for fit in (result['fits'][0], result['inverseFits'][0]):
            self.assertEqual(fit['observedX'], [11,13,14,15,16,17,18,19,20])
        self.assertEqual(result['series'][0]['x'], [15.5])
        options.update(trialStart=15, trialEnd=16)
        short = analyze(rows, options)
        self.assertEqual(short['series'][0]['x'], [15.5])
        self.assertFalse(short['fits'][0]['ok'])
        self.assertFalse(short['inverseFits'][0]['ok'])
        options.update(trialStart=30, trialEnd=40)
        empty = analyze(rows, options)
        self.assertEqual(empty['included'], 0)
        self.assertEqual(empty['fits'], [])
        self.assertEqual(empty['inverseFits'], [])
        self.assertEqual(empty['summary'], [])

    def test_trial_range_validation(self):
        self.assertEqual(trial_range({}), (1, None))
        self.assertEqual(trial_range(dict(trialStart=100, trialEnd=100)), (100,100))
        for start,end in [(0,10), (1,0), (10,5), (1.5,10), (1,10.5), (True,10), (1,False)]:
            with self.subTest(start=start,end=end), self.assertRaises(ValueError):
                trial_range(dict(trialStart=start,trialEnd=end))

    def test_partial_range_preserves_bin_boundaries(self):
        rows = [dict(participant=p, n=n, success=True, phase='post', condition='C1',
                     errorType='none', region='none', coreMtMs=dict(MT=n, ER=0, TPe=n))
                for p in ('001','002') for n in range(1,31)]
        options = dict(scope='coreMtMs', phase='post', mode='mean', bin=10,
                       participants=['001','002'], conditions=['C1'], exclude=False, trialStart=8, trialEnd=13)
        result=analyze(rows, options)
        self.assertEqual(result['included'], 12)
        self.assertEqual(result['series'][0]['x'], [9,12])
        self.assertEqual(result['series'][0]['MT'], [9,12])
        options['trialEnd']=None
        self.assertEqual(analyze(rows,options)['included'],46)

    def test_boundary_windows_are_optional_and_not_double_counted(self):
        rows = [dict(participant='001', n=n, success=n != 100, phase='post', condition='C1',
                     source='test.csv', line=n+1, attempt=n, errorType='none', region='none',
                     coreMtMs=dict(MT=n, ER=0, TPe=1)) for n in range(1, 311)]
        options = dict(scope='coreMtMs', phase='post', mode='individual', bin=10,
                       participants=['001'], conditions=['C1'], exclude=True,
                       excludeBoundaryTrials=True, trialStart=1, trialEnd=None)
        result = analyze(rows, options)
        self.assertEqual(result['selected'], 310)
        self.assertEqual(result['included'], 280)
        self.assertEqual(len(result['excluded']), 30)
        self.assertEqual(result['summary'][0]['rows'], 280)
        overlap = next(row for row in result['excluded'] if row['n'] == 100)
        self.assertEqual(overlap['reason'], 'success=false / 100・200・300試行周辺10試行')
        options['excludeBoundaryTrials'] = False
        kept = analyze(rows, options)
        self.assertEqual(kept['included'], 309)
        self.assertEqual(len(kept['excluded']), 1)
        options.update(excludeBoundaryTrials=True, trialStart=100, trialEnd=200)
        partial = analyze(rows, options)
        self.assertEqual(partial['selected'], 101)
        self.assertEqual(partial['included'], 90)
        self.assertEqual(len(partial['excluded']), 11)

    def test_trial_number_mt_outlier_uses_three_sample_sd(self):
        rows = []
        for i in range(12):
            mt = 1000 if i == 11 else 100
            rows.append(dict(participant=f'{i:03}', n=1, success=True, phase='post', condition='C1',
                             source='test.csv', line=i+2, attempt=1, errorType='none', region='none',
                             coreMtMs=dict(MT=mt, ER=0, TPe=1)))
        found = mt_outliers(rows, 'coreMtMs', 'trial3sd')
        self.assertEqual([row['participant'] for row in rows if id(row) in found], ['011'])
        options = dict(scope='coreMtMs', phase='post', mode='mean', bin=10,
                       participants=[row['participant'] for row in rows], conditions=['C1'], exclude=False,
                       mtOutlier='trial3sd')
        result = analyze(rows, options)
        self.assertEqual(result['included'], 11)
        self.assertEqual(result['excluded'][0]['reason'], '同一試行番号MTが平均±3SD超過')
        self.assertEqual(result['summary'][0]['ER'], 0)

    def test_learning_curve_residual_mt_outlier(self):
        rows = []
        for n in range(1, 41):
            mt = float(curve(n, [300, 1500, 4, .6])) + (5000 if n == 20 else 0)
            rows.append(dict(participant='001', n=n, success=True, phase='practice', condition='C1',
                             source='test.csv', line=n+1, attempt=1, errorType='none', region='none',
                             coreMtMs=dict(MT=mt, ER=0, TPe=1)))
        found = mt_outliers(rows, 'coreMtMs', 'curve3sd')
        self.assertEqual([row['n'] for row in rows if id(row) in found], [20])
        options = dict(scope='coreMtMs', phase='practice', mode='individual', bin=10,
                       participants=['001'], conditions=['C1'], exclude=False, mtOutlier='curve3sd')
        result = analyze(rows, options)
        self.assertEqual(result['included'], 39)
        self.assertEqual(result['excluded'][0]['reason'], '習熟曲線MT残差が平均±3SD超過')
        self.assertNotIn(20, result['fits'][0]['observedX'])


if __name__ == '__main__':
    unittest.main()
