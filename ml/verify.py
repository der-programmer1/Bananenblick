#!/usr/bin/env python3
"""Meaningful checks for transforms, metric semantics and split integrity."""
from pathlib import Path
import json,unittest,tempfile
from unittest.mock import patch
import numpy as np
from train import prepare,metrics,iou,BananaDataset
from scene_augmentation import visible_targets
from prepare_data import normalize_display_image
from PIL import Image
ROOT=Path(__file__).resolve().parent

class PipelineChecks(unittest.TestCase):
    def test_visible_boxes_shrink_after_occlusion(self):
        first=np.zeros((20,20),bool);first[:,:10]=True
        second=np.zeros((20,20),bool);second[:,10:]=True
        targets=visible_targets([first,second],[1,0],[400,200])
        np.testing.assert_array_equal(targets,[[1,5,10,10,20],[0,15,10,10,20]])

    def test_almost_hidden_training_fruit_is_not_given_a_full_box(self):
        mask=np.zeros((20,20),bool);mask[:,:1]=True
        self.assertEqual(len(visible_targets([mask],[1],[400])),0)

    def test_horizontal_flip_uses_the_selected_model_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);Image.new('RGB',(64,64),'yellow').save(root/'image.png')
            (root/'train.json').write_text(json.dumps([{'id':0,'file':'image.png','boxes':[[1,.25,.25,.5,.5]]}]))
            ds=BananaDataset(root,'train',augment=True,mosaic=0,size=448)
            with patch('train.random.random',return_value=0),patch('train.random.uniform',side_effect=[1,0,1,1]):
                image,targets=ds[0]
            self.assertEqual(tuple(image.shape),(3,448,448))
            np.testing.assert_allclose(targets[0],[1,336,112,224,224])

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
        r=metrics([[p]],[[{'box':[0,0,100,100],'class':0}]],classification=.6,margin=.15)
        self.assertEqual(r['recall'],0)
        self.assertEqual(r['perClass'][0]['wrongRipeness'],1)
        self.assertEqual(r['perClass'][0]['localizationRecall'],1)
        self.assertEqual(r['perClass'][1]['falsePositive'],1)
    def test_negative_frame_false_alarm_is_counted(self):
        p={'box':[0,0,100,100],'class':1,'score':.9,'classProbability':.9,'classMargin':.8}
        r=metrics([[p]],[[]]);self.assertEqual(r['perClass'][1]['falsePositive'],1)
    def test_low_confidence_is_uncertain(self):
        p={'box':[0,0,100,100],'class':0,'score':.4,'classProbability':.5,'classMargin':.1}
        r=metrics([[p]],[[{'box':[0,0,100,100],'class':0}]],classification=.6,margin=.15)
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
