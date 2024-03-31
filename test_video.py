import os
import sys
import cv2
import torch
import numpy as np
from config import Config
from models.magnet import MagNet
from data import get_gen_ABC, unit_postprocessing, numpy2cuda, resize2d
from callbacks import gen_state_dict
from PIL import Image

# config
config = Config()
# Load weights
ep = ''

weights_files = sorted(
    [p for p in os.listdir(config.save_dir) if '_loss' in p and 'D' not in p],
    key=lambda x: float(x.rstrip('.pth').split('_loss')[-1])
)

for weights_file in weights_files:
    weights_path = os.path.join(config.save_dir, weights_file)
    epo = int(weights_file.split('_epoch')[1].split('_')[0])
    if  epo ==int(99):  # TODO: choose epoch No.
        state_dict = gen_state_dict(weights_path)
        model_test = MagNet().cuda()
        model_test.load_state_dict(state_dict, strict=False)
        model_test.eval()
        print("Loading weights:", weights_file)
        if len(sys.argv) == 1:
            testsets = 'baby'  # TODO: choose video name
        else:
            testsets = sys.argv[-1]
        testsets = testsets.split('-')
        dir_results = config.save_dir.replace('weights', 'all_results')
        if not os.path.exists(dir_results):
            os.makedirs(dir_results)
        for testset in testsets:
            data_loader = get_gen_ABC(config, mode='test_on_'+testset)
            print('Number of test image couples:', data_loader.data_len)
            img = cv2.imread(data_loader.paths[0])
            vid_size = img.shape[:2][::-1]
            # Test 
            for amp in [20]:
                frames = []
                frame = []  ###

                video_name_epo= testset + '_epoch{}_amp{}'.format(epo,amp)
                video_dir = os.path.join(dir_results, video_name_epo)
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)
                for idx_load in range(0, data_loader.data_len, data_loader.batch_size):
                    if (idx_load+1) % 100 == 0:
                        print('{}'.format(idx_load+1), end=', ')
                    batch_A, batch_B = data_loader.gen_test0()
                    amp_factor = numpy2cuda(amp)
                    frame.append(batch_A)
                    frame0 = frame[0]
                    for _ in range(len(batch_A.shape) - len(amp_factor.shape)):
                        amp_factor = amp_factor.unsqueeze(-1)
                    with torch.no_grad():
                        # y_hats = model_test(batch_A, batch_B, amp_factor , 0 , mode='evaluate')  # Dynamic mode
                        y_hats = model_test(frame0, batch_B, amp_factor , 0 , mode='evaluate')  # Static mode
                    for y_hat in y_hats:
                        y_hat = unit_postprocessing(y_hat, vid_size=vid_size)
                        frames.append(y_hat)
                        if len(frames) >= data_loader.data_len:
                            break
                    if len(frames) >= data_loader.data_len:
                        break
                data_loader = get_gen_ABC(config, mode='test_on_'+testset)
                frames = [unit_postprocessing(data_loader.gen_test0()[0], vid_size=vid_size)] + frames
                # Make videos of framesMag
                img_dir = os.path.join(video_dir, 'img_{}_amp{}'.format(testset, amp))
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                for i, frame in enumerate(frames):
                    fn = os.path.join(img_dir, 'img_{}_{}.png'.format(testset, i))
                    im = Image.fromarray(frame)
                    im.save(fn)

                FPS = 30
                out = cv2.VideoWriter(
                    os.path.join(video_dir, '{}_amp{}_epoch{}.avi'.format(testset, amp, epo)),
                    cv2.VideoWriter_fourcc(*'DIVX'),
                    FPS, frames[0].shape[-2::-1]
                )
                for frame in frames:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame)
                out.release()
                print('{} has been done.'.format(os.path.join(video_dir, '{}_amp{}_epoch{}.avi'.format(testset, amp, epo))))

