import json
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from run_baselines import ROOT,FEATURES,LABELS,load_data,sha256
from feature_models import PERMITTED_INPUTS,DERIVED,feature_frame,FeatureSVM,InteractionLogistic
from run_feature_ablation import configurations


class FeatureAblationTests(unittest.TestCase):
    def test_ratios_logs_and_exclusions(self):
        X=pd.DataFrame({"daily_short_form_hours":[2.],"avg_clip_duration_sec":[6.],
            "swipes_per_minute":[10.],"scroll_sessions_day":[4.],"bedtime_scroll_min":[30.],
            "total_screen_time_hours":[5.],"brain_fog_score":[999.],"cognitive_decay_tier":[LABELS[0]]})
        frame=feature_frame(X,'engineered_log')
        np.testing.assert_allclose(frame[DERIVED].iloc[0],[30,1,.25,.4])
        np.testing.assert_allclose(frame['log1p_daily_short_form_hours'],np.log1p(2))
        self.assertEqual(len(frame.columns),16)
        self.assertNotIn('brain_fog_score',frame)
        self.assertNotIn('cognitive_decay_tier',frame)
        X.loc[0,'scroll_sessions_day']=0
        with self.assertRaises(ValueError):feature_frame(X,'engineered')

    def test_preprocessing_uses_fitted_training_categories(self):
        data=load_data(ROOT/'short_form_brain_rot_cognitive_decay.csv').iloc[:200]
        X=data[PERMITTED_INPUTS];y=data.cognitive_decay_tier
        with threadpool_limits(limits=1):model=FeatureSVM(variant='demographics_platform').fit(X,y)
        preprocessing=model.model_.named_steps['columntransformer']
        np.testing.assert_allclose(preprocessing.named_transformers_['numeric'].mean_,X[FEATURES+['age']].mean())
        before=model.output_feature_names_.copy()
        new=X.iloc[:3].copy();new['primary_platform']='unseen-platform'
        self.assertEqual(len(model.predict(new)),3)
        self.assertEqual(model.output_feature_names_,before)

    def test_interaction_dimensions_and_training_only_scaling(self):
        data=load_data(ROOT/'short_form_brain_rot_cognitive_decay.csv').iloc[:200]
        with threadpool_limits(limits=1):
            for interaction,size in [(False,6),(True,21)]:
                model=InteractionLogistic(interactions=interaction).fit(data[PERMITTED_INPUTS],data.cognitive_decay_tier)
                self.assertEqual(len(model.output_feature_names_),size)
                self.assertEqual(model.raw_input_features_,FEATURES)
                np.testing.assert_allclose(model.model_.steps[0][1].mean_,data[FEATURES].mean())

    def test_matched_search_budgets(self):
        settings=configurations()
        self.assertEqual(len(settings),6)
        for values in settings.values():self.assertEqual(len(values),12)
        baseline=[{k:v for k,v in p.items() if k!='variant'} for p in settings['svm_context']]
        for name in ['svm_demographics_platform','svm_engineered','svm_engineered_log']:
            self.assertEqual(baseline,[{k:v for k,v in p.items() if k!='variant'} for p in settings[name]])

    @unittest.skipUnless((ROOT/'results/feature_ablation/RESULTS.md').exists(),'Feature study not complete')
    def test_saved_evidence(self):
        out=ROOT/'results/feature_ablation'
        m=json.loads((out/'manifest.json').read_text())
        self.assertEqual(m['status'],'complete')
        self.assertFalse(m['test_evaluated'])
        self.assertEqual(m['script_sha256'],sha256(ROOT/'run_feature_ablation.py'))
        self.assertEqual(m['model_script_sha256'],sha256(ROOT/'feature_models.py'))
        original=json.loads((ROOT/'results/baselines/42/splits.json').read_text())
        allowed=set(original['train_row_positions']);excluded=set(original['test_row_positions'])
        for name in ['original_svm',*configurations()]:
            pred=pd.read_csv(out/f'{name}_oof_predictions.csv')
            self.assertEqual(len(pred),7200)
            self.assertTrue(pred.row_position.is_unique)
            self.assertEqual(set(pred.row_position),allowed)
            self.assertFalse(set(pred.row_position)&excluded)
            value=f1_score(pred.true_label,pred.predicted_label,labels=LABELS,average='macro',zero_division=0)
            self.assertAlmostEqual(value,json.loads((out/f'{name}_pooled_metrics.json').read_text())['macro_f1'])
        for fold in range(3):
            folder=out/f'fold_{fold}'
            self.assertEqual(json.loads((folder/'splits.json').read_text()),json.loads(
                (ROOT/f'results/tuning/fold_{fold}/splits.json').read_text()))
            for name in configurations():
                self.assertEqual(len(pd.read_csv(folder/name/'cv_results.csv')),12)
                info=json.loads((folder/name/'selection.json').read_text())
                self.assertTrue(set(info['raw_input_features_']).issubset(set(PERMITTED_INPUTS)))


if __name__=='__main__':unittest.main()
