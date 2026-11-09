#!/usr/bin/env python3
"""Meaningful checks for transforms, metric semantics and split integrity."""
from pathlib import Path
import json,unittest
import numpy as np
from train import prepare,metrics,iou
from prepare_data import normalize_display_image
from PIL import Image
ROOT=Path(__file__).resolve().parent

class PipelineChecks(unittest.TestCase):
    def test_exif_orientation_is_applied_before_resize(self):
        im=Image.new('RGB',(3,2),'black');im.putpixel((0,0),(255,0,0));im.getexif()[274]=6
        shown=normalize_display_image(im)
        self.assertEqual(shown.size,(2,3))
        self.assertEqual(shown.getpixel((1,0)),(255,0,0))

    def test_top_left_letterbox_coordinates(self):
        im=np.zeros((100,200,3),np.uint8)
        canvas,boxes=prepare(im,[[1,.5,.5,.5,.5]])
        self.assertEqual(canvas.shape,(320,320,3))
        np.testing.assert_allclose(boxes[0],[1,160,80,160,80])
        self.assertTrue(np.all(canvas[160:]==114))
    def test_iou_and_wrong_class_are_not_detection_success(self):
        p={'box':[0,0,100,100],'class':1,'score':.9,'classProbability':.9,'classMargin':.8}
        r=metrics([[p]],[[{'box':[0,0,100,100],'class':0}]])
        self.assertEqual(r['recall'],0)
        self.assertEqual(r['perClass'][0]['wrongRipeness'],1)
        self.assertEqual(r['perClass'][0]['localizationRecall'],1)
        self.assertEqual(r['perClass'][1]['falsePositive'],1)
    def test_negative_frame_false_alarm_is_counted(self):
        p={'box':[0,0,100,100],'class':1,'score':.9,'classProbability':.9,'classMargin':.8}
        r=metrics([[p]],[[]]);self.assertEqual(r['perClass'][1]['falsePositive'],1)
    def test_low_confidence_is_uncertain(self):
        p={'box':[0,0,100,100],'class':0,'score':.4,'classProbability':.5,'classMargin':.1}
        r=metrics([[p]],[[{'box':[0,0,100,100],'class':0}]])
        self.assertEqual(r['visibleConfusion']['counts'][0],[0,0,0,1,0])
    def test_actual_split_integrity(self):
        data=ROOT/'work/data'
        if not (data/'test.json').exists():self.skipTest('Prepare public data first')
        sets={s:json.loads((data/f'{s}.json').read_text()) for s in ['train','val','test']}
        for a,b in [('train','val'),('train','test'),('val','test')]:
            self.assertFalse({r['sha256_pixels'] for r in sets[a]}&{r['sha256_pixels'] for r in sets[b]})
            self.assertFalse({r['group'] for r in sets[a]}&{r['group'] for r in sets[b]})
            for left in sets[a]:
                for right in sets[b]:
                    self.assertGreater(min((int(x)^int(y)).bit_count() for x in left['phash'] for y in right['phash']),6)
        for records in sets.values():
            self.assertTrue(all(any(int(box[0])==c for r in records for box in r['boxes']) for c in range(3)))

if __name__=='__main__':unittest.main()
