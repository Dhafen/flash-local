import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from compare_model_quality import BASELINE, BASELINE_REVISION, CATEGORIES, compare


def report(candidate=False):
    return {'schemaVersion': 1, 'inferenceLocation': 'local_iPhone' if candidate else 'local_PC',
            'model': {'upstreamRepository': 'Akicou/Qwen3.8-Flash-Next-REAM-60Pct' if candidate else BASELINE,
                      'upstreamRevision': 'c' * 40 if candidate else BASELINE_REVISION,
                      'artifactSHA256': ('a' if candidate else 'b') * 64, 'quantization': 'declared fixture precision'},
            'runtimeSourceRevision': 'd' * 40, 'datasetSHA256': 'e' * 64, 'graderSHA256': 'f' * 64,
            'generationPolicy': {'thinking': 'off', 'contextTokens': 1024, 'maximumNewTokens': 128,
                                 'sampling': {'kind': 'greedy', 'seed': 0}},
            'tasks': [{'id': category + '-fixture', 'category': category, 'promptSHA256': '1' * 64,
                       'passed': True, 'checksExecuted': 2, 'finishReason': 'stop'} for category in sorted(CATEGORIES)]}


class QualityComparisonTests(unittest.TestCase):
    def testPairedPassesDoNotClaimGeneralEquivalenceAuthenticityOrSpeed(self):
        result = compare(report(), report(True))
        self.assertTrue(result['meetsObservedTaskFloor'])
        for key in ['generalQualityEquivalenceEstablished', 'reportAuthenticityVerified', 'speedComparisonPerformed']:
            self.assertFalse(result[key])

    def testRegressionInAnyCategoryFailsEvenIfOtherCategoriesPass(self):
        candidate = report(True); candidate['tasks'][0]['passed'] = False
        result = compare(report(), candidate)
        self.assertFalse(result['meetsObservedTaskFloor'])
        self.assertEqual(result['regressedTaskIDs'], [candidate['tasks'][0]['id']])

    def testOmittedDuplicatedUnscoredOrDifferentPromptTasksAreRefused(self):
        for change in [lambda r: r['tasks'].pop(),
                       lambda r: r['tasks'].append(copy.deepcopy(r['tasks'][0])),
                       lambda r: r['tasks'][0].pop('passed'),
                       lambda r: r['tasks'][0].update(promptSHA256='2' * 64)]:
            candidate = report(True); change(candidate)
            with self.assertRaises(ValueError): compare(report(), candidate)

    def testDifferentThinkingSamplingBudgetDatasetOrGraderAreRefused(self):
        for change in [lambda r: r['generationPolicy'].update(thinking='xhigh'),
                       lambda r: r['generationPolicy']['sampling'].update(seed=1),
                       lambda r: r['generationPolicy'].update(maximumNewTokens=256),
                       lambda r: r.update(datasetSHA256='2' * 64),
                       lambda r: r.update(graderSHA256='2' * 64)]:
            candidate = report(True); change(candidate)
            with self.assertRaises(ValueError): compare(report(), candidate)

    def testTruncatedOrUntestedOutputsCannotPassButCountAsFailures(self):
        for key, value in [('finishReason', 'length'), ('checksExecuted', 0)]:
            candidate = report(True); candidate['tasks'][0][key] = value
            with self.assertRaises(ValueError): compare(report(), candidate)
            candidate['tasks'][0]['passed'] = False
            self.assertFalse(compare(report(), candidate)['meetsObservedTaskFloor'])

    def testCloudRunsUnknownWeightsAndWrongBaselineAreRefused(self):
        for change in [lambda r: r.update(inferenceLocation='cloud'),
                       lambda r: r['model'].update(artifactSHA256=None),
                       lambda r: r['model'].update(upstreamRepository='Qwen/Qwen3.6-27B')]:
            baseline = report(); change(baseline)
            with self.assertRaises(ValueError): compare(baseline, report(True))

    def testNonfinitePoliciesAndBooleanCountsAreRefused(self):
        for change in [lambda r: r['generationPolicy'].update(sampling={
                           'kind': 'categorical', 'seed': 0, 'temperature': float('nan'), 'topP': 0.95, 'topK': 20}),
                       lambda r: r['tasks'][0].update(checksExecuted=True),
                       lambda r: r['generationPolicy']['sampling'].update(seed=True),
                       lambda r: r['generationPolicy'].update(contextTokens=True)]:
            candidate = report(True); change(candidate)
            with self.assertRaises(ValueError): compare(report(), candidate)

    def testCategoricalSamplingIsExplicitAndMustMatch(self):
        baseline, candidate = report(), report(True)
        policy = {'kind': 'categorical', 'seed': 123, 'temperature': 1.0, 'topP': 0.95, 'topK': 20}
        for record in [baseline, candidate]:
            record['generationPolicy']['sampling'] = copy.deepcopy(policy)
        self.assertTrue(compare(baseline, candidate)['meetsObservedTaskFloor'])
        for key, value in [('temperature', 0), ('topP', 1.1), ('topK', -1), ('seed', -1), ('topP', True)]:
            changed = copy.deepcopy(candidate); changed['generationPolicy']['sampling'][key] = value
            with self.assertRaises(ValueError): compare(baseline, changed)

    def testUnknownGenerationOrSamplingFieldsAreRefused(self):
        for change in [lambda r: r['generationPolicy'].update(privatePrompt='unrecognized data'),
                       lambda r: r['generationPolicy']['sampling'].update(privatePrompt='unrecognized data'),
                       lambda r: r['generationPolicy']['sampling'].update(kind='unspecified'),
                       lambda r: r['generationPolicy']['sampling'].pop('seed')]:
            baseline, candidate = report(), report(True)
            change(baseline); change(candidate)
            with self.assertRaises(ValueError): compare(baseline, candidate)

    def testExtraConversationFieldsAreNotCopiedIntoTheAggregate(self):
        candidate = report(True)
        candidate['model']['privateConversation'] = 'not an identity field'
        candidate['tasks'][0]['responseText'] = 'private generated code'
        result = compare(report(), candidate)
        self.assertNotIn('privateConversation', result['candidate'])
        self.assertNotIn('private generated code', str(result))
        self.assertEqual(result['executions']['candidate'], {
            'inferenceLocation': 'local_iPhone', 'runtimeSourceRevision': 'd' * 40})


if __name__ == '__main__':
    unittest.main()
