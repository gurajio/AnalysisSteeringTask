import math
import unittest
from unittest.mock import patch

import numpy as np
from analysis import aggregate, analyze, clip_points, core_ip, curve, event_region, fit_curve, mt_outliers, trajectory_metrics, trial_range


class AnalysisTests(unittest.TestCase):
    def test_core_exit_filter(self):
        rows = [dict(participant='001', n=i+1, success=False, countedAsTrial=counted,
                     coreExitMs=exit_time, phase='practice', condition='C2',
                     source='test.csv', line=i+2, attempt=i+1, errorType='deviation', region='inside',
                     IP=10, coreMtMs=dict(MT=100, ER=1, TPe=2), mtMs=dict(MT=200, ER=1, TPe=4))
                for i, (counted, exit_time) in enumerate([(True, 1000), (True, None),
                                                          (False, 1000), (False, None), (True, 0)])]
        options = dict(scope='coreMtMs', phase='practice', mode='mean', bin=10,
                       participants=['001'], conditions=['C2'], exclude=True)
        for scope in ('coreMtMs', 'mtMs'):
            options['scope'] = scope
            result = analyze(rows, options)
            self.assertEqual(result['included'], 2)
            self.assertEqual([r['reason'] for r in result['excluded']],
                             ['coreExitMs欠損', 'countedAsTrial=false', 'countedAsTrial=false / coreExitMs欠損'])
            self.assertEqual(result['summary'][0]['rows'], 2)
            self.assertEqual(result['regions']['inside'], 2)
            self.assertEqual(result['series'][0]['IP'], [10])
            for fit in (result['fits'][0], result['inverseFits'][0]):
                self.assertEqual(fit['observedX'], [1, 5])
        options['exclude'] = False
        self.assertEqual(analyze(rows, options)['included'], 5)
        options.update(exclude=True, trialStart=1, trialEnd=1)
        self.assertEqual(analyze(rows, options)['excluded'], [])

    def test_counted_failure_is_included(self):
        rows = [dict(coreExitMs=1000, participant='001', n=1, success=success, countedAsTrial=counted,
                     phase='post', condition='C2', source='test.csv', line=i+2, attempt=i+1,
                     errorType=error, region='none', coreMtMs=dict(MT=mt, ER=0, TPe=1))
                for i, (success, counted, error, mt) in enumerate([
                    (True, True, 'none', 100),
                    (False, True, 'near_goal_pointerup', 300),
                    (False, False, 'pointerup_mid_path', 800)])]
        options = dict(scope='coreMtMs', phase='post', mode='mean', bin=10,
                       participants=['001'], conditions=['C2'], exclude=True)
        result = analyze(rows, options)
        self.assertEqual(result['included'], 2)
        self.assertEqual(result['summary'][0]['MT'], 200)
        self.assertEqual(result['excluded'][0]['reason'], 'countedAsTrial=false')
        options['exclude'] = False
        self.assertEqual(analyze(rows, options)['included'], 3)

    def test_core_ip_units(self):
        self.assertAlmostEqual(core_ip(1000, 1350, 70), 1000 / (1350 / 70))
        for values in [(None, 10, 2), (100, 0, 2), (100, 10, -1), (100, 'nan', 2), (0, 10, 2)]:
            self.assertIsNone(core_ip(*values))

    def test_ip_trend_scope_and_filters(self):
        rows = []
        for p, n, countedAsTrial, ip in [('001', 1, True, 10), ('001', 2, True, 30),
                                   ('002', 1, True, 100), ('002', 2, False, 900),
                                   ('002', 3, True, None), ('002', 96, True, 800)]:
            rows.append(dict(coreExitMs=1000, participant=p, n=n, countedAsTrial=countedAsTrial, phase='post', condition='C1',
                             source='test.csv', line=n, attempt=n, errorType='none', region='none', IP=ip,
                             coreMtMs=dict(MT=100, ER=0, TPe=1), mtMs=dict(MT=200, ER=0, TPe=2)))
        options = dict(scope='mtMs', phase='post', mode='mean', bin=10, participants=['001', '002'],
                       conditions=['C1'], exclude=True, excludeBoundaryTrials=True, mtOutlier='none')
        result = analyze(rows, options)
        self.assertEqual(result['series'][0]['IP'], [60])
        self.assertEqual(result['missing']['IP'], 1)
        self.assertEqual(result['summary'][0]['IPParticipants'], 2)
        options['scope'] = 'coreMtMs'
        self.assertEqual(analyze(rows, options)['series'][0]['IP'], [60])
        options.update(trialStart=2, trialEnd=3)
        result = analyze(rows, options)
        self.assertEqual(result['series'][0]['IP'], [30])
        self.assertEqual(result['series'][0]['x'], [2.5])
        options.update(mode='individual', trialStart=1, trialEnd=None)
        self.assertEqual([line['IP'] for line in analyze(rows, options)['series']], [[20], [100]])

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
        for p, n, countedAsTrial, mt, er in [('001', 1, True, 100, 1), ('001', 2, True, 300, 0),
                                     ('002', 1, True, 1000, 0), ('002', 2, False, 5000, 0)]:
            rows.append(dict(coreExitMs=1000, participant=p, n=n, countedAsTrial=countedAsTrial, phase='post', condition='C1',
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

    def test_power_curve_units(self):
        x = np.arange(1, 41, dtype=float)
        for unit in (1, 1000):
            expected = [300 / unit, 1500 / unit, .6]
            y = curve(x, expected)
            result = fit_curve(x, y, unit_scale=unit, model='power')
            self.assertTrue(result['ok'])
            self.assertEqual(result['parameters']['E'], 0)
            np.testing.assert_allclose([result['parameters'][k] for k in ('A', 'B', 'a')], expected, rtol=1e-5)
            self.assertLess(result['rmse'], .01 / unit)
        self.assertFalse(fit_curve(x[:7], y[:7], model='power')['ok'])

    def test_unrestricted_a(self):
        x = np.arange(1, 101, dtype=float)
        for model in ('shifted', 'power'):
            for unit in (1, 1000):
                params = [-100 / unit, 2000 / unit, 4. if model == 'shifted' else 0., .5]
                y = curve(x, params)
                free = fit_curve(x, y, unit_scale=unit, model=model, a_bound='free')
                bounded = fit_curve(x, y, unit_scale=unit, model=model)
                self.assertTrue(free['ok'])
                self.assertAlmostEqual(free['parameters']['A'], params[0], delta=.01 / unit)
                self.assertLess(free['rmse'], .001 / unit)
                self.assertGreaterEqual(bounded['parameters']['A'], 0)
                self.assertLess(free['rmse'], bounded['rmse'])

    def test_fixed_a_recovery(self):
        x = np.arange(1, 101, dtype=float)
        for model in ('shifted', 'power'):
            for value in (84., 253., 315.):
                params = [value, 1500., 4. if model == 'shifted' else 0., .6]
                result = fit_curve(x, curve(x, params), model=model, fixed_a=value)
                self.assertTrue(result['ok'])
                self.assertTrue(result['aFixed'])
                self.assertEqual(result['parameters']['A'], value)
                self.assertLess(result['rmse'], .001)
                np.testing.assert_allclose(result['y'], curve(result['x'], params), atol=.001)

    def test_fixed_core_scope(self):
        rows = [dict(coreExitMs=1000, participant='001', n=n, countedAsTrial=True, phase='practice', condition=c,
                     source='test.csv', line=n+1, attempt=n, errorType='none', region='none',
                     coreMtMs=dict(MT=float(curve(n, [a, 1500., .6])), ER=0, TPe=2),
                     mtMs=dict(MT=float(curve(n, [500., 1500., .6])), ER=0, TPe=2))
                for c, a in [('C1', 84.), ('C2', 253.), ('C3', 315.)] for n in range(1, 41)]
        options = dict(scope='coreMtMs', phase='practice', mode='mean', bin=10,
                       participants=['001'], conditions=['C1', 'C2', 'C3'], exclude=False,
                       curveModel='power', aBound='fixed_core', mtOutlier='none')
        result = analyze(rows, options)
        self.assertEqual([f['parameters']['A'] for f in result['fits']], [84., 253., 315.])
        self.assertTrue(all(f['aFixed'] for f in result['fits']))
        self.assertTrue(all(not f['aFixed'] for f in result['inverseFits']))
        for scope in ('coreMtMs', 'mtMs'):
            for method in ('curve3sd', 'curve1_5sd'):
                with patch('analysis.fit_curve', return_value=dict(ok=False)) as mock:
                    mt_outliers(rows, scope, method, model='power', a_bound='fixed_core')
                expected = [84., 253., 315.] if scope == 'coreMtMs' else [None]*3
                self.assertEqual([c.kwargs['fixed_a'] for c in mock.call_args_list], expected)
        options['scope'] = 'mtMs'
        result = analyze(rows, options)
        self.assertTrue(all(not f['aFixed'] for f in result['fits'] + result['inverseFits']))
        self.assertAlmostEqual(result['fits'][0]['parameters']['A'], 500., places=2)

    def test_selected_model_propagation(self):
        rows = [dict(coreExitMs=1000, participant='001', n=n, countedAsTrial=True, phase='practice', condition='C1',
                     source='test.csv', line=n+1, attempt=1, errorType='none', region='none',
                     coreMtMs=dict(MT=float(curve(n, [300., 1500., .6])), ER=0, TPe=2)) for n in range(1, 41)]
        options = dict(scope='coreMtMs', phase='practice', mode='individual', bin=10,
                       participants=['001'], conditions=['C1'], exclude=False, curveModel='power', aBound='free')
        for method in ('curve3sd', 'curve1_5sd'):
            options['mtOutlier'] = method
            with patch('analysis.fit_curve', wraps=fit_curve) as mock:
                result = analyze(rows, options)
                self.assertEqual(mock.call_count, 3)
                self.assertTrue(all(call.kwargs['model'] == 'power' for call in mock.call_args_list))
                self.assertTrue(all(call.kwargs['a_bound'] == 'free' for call in mock.call_args_list))
                self.assertEqual(mock.call_args_list[-1].kwargs['unit_scale'], 1000)
                self.assertEqual(result['fits'][0]['parameters']['E'], 0)

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
            rows.append(dict(coreExitMs=1000, participant=p, n=1, countedAsTrial=True, phase='practice', condition='C1',
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
        rows = [dict(coreExitMs=1000, participant='001', n=i + 1, countedAsTrial=i != 1, phase='practice', condition='C1',
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
        rows = [dict(coreExitMs=1000, participant='001', n=n, countedAsTrial=n not in (2, 12), phase='practice', condition='C1',
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
        rows = [dict(coreExitMs=1000, participant=p, n=n, countedAsTrial=True, phase='post', condition='C1',
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
        rows = [dict(coreExitMs=1000, participant='001', n=n, countedAsTrial=n != 100, phase='post', condition='C1',
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
        self.assertEqual(overlap['reason'], 'countedAsTrial=false / 100・200・300試行周辺10試行')
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
            rows.append(dict(coreExitMs=1000, participant=f'{i:03}', n=1, countedAsTrial=True, phase='post', condition='C1',
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

    def test_trial_one_half_sd(self):
        rows = [dict(coreExitMs=1000, participant=str(i), n=1, phase='post', condition='C1', countedAsTrial=True,
                     source='test.csv', line=i+2, attempt=1, errorType='none', region='none',
                     coreMtMs=dict(MT=mt, ER=0, TPe=1)) for i, mt in enumerate([100, 100, 100, 200])]
        self.assertEqual(mt_outliers(rows, 'coreMtMs', 'trial1_5sd'), set())
        rows.append(dict(rows[0], participant='4'))
        self.assertEqual(mt_outliers(rows, 'coreMtMs', 'trial1_5sd'), {id(rows[3])})
        self.assertEqual(mt_outliers(rows, 'coreMtMs', 'trial3sd'), set())
        options = dict(scope='coreMtMs', phase='post', mode='mean', bin=10,
                       participants=[r['participant'] for r in rows], conditions=['C1'], exclude=False,
                       mtOutlier='trial1_5sd')
        result = analyze(rows, options)
        self.assertEqual(result['included'], 4)
        self.assertEqual(result['excluded'][0]['reason'], '同一試行番号MTが平均±1.5SD超過')

    def test_residual_one_half_sd(self):
        rows = [dict(coreExitMs=1000, participant='001', n=i+1, condition='C1',
                     coreMtMs=dict(MT=mt)) for i, mt in enumerate([100]*7 + [200])]
        fit = dict(ok=True, parameters=dict(A=100., B=0., E=0., a=1.))
        with patch('analysis.fit_curve', return_value=fit) as mock:
            self.assertEqual(mt_outliers(rows, 'coreMtMs', 'curve1_5sd'), {id(rows[-1])})
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(mt_outliers(rows, 'coreMtMs', 'curve3sd'), set())

    def test_learning_curve_residual_mt_outlier(self):
        rows = []
        for n in range(1, 41):
            mt = float(curve(n, [300, 1500, 4, .6])) + (5000 if n == 20 else 0)
            rows.append(dict(coreExitMs=1000, participant='001', n=n, countedAsTrial=True, phase='practice', condition='C1',
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
