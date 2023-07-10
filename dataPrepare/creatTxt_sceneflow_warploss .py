import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(leftinputdir), 'Input dir not found'
    assert os.path.exists(rightinputdir), 'Input dir not found'

    assert os.path.exists(lefttargetdir), 'target dir not found'
    assert os.path.exists(righttargetdir), 'target dir not found'

    mkdir(outputdir)
    leftinput = os.listdir(leftinputdir)
    leftinput.sort()
    for vid in leftinput:
        print(vid)
        frames = os.listdir(os.path.join(leftinputdir, vid))
        print(len(frames))

        for idx in range(0, len(frames)):
            groups = ''
            if idx == 0:
                groups += os.path.join(leftinputdir, vid, '00000' + ext) + '|'  # 0
                groups += os.path.join(leftinputdir, vid, '00000' + ext) + '|'  # 1
                groups += os.path.join(leftinputdir, vid, '00001' + ext) + '|'  # 2
                groups += os.path.join(leftinputdir, vid, '00002' + ext) + '|'   # 3
                groups += os.path.join(leftinputdir, vid, '00003' + ext) + '|'  # 4

                groups += os.path.join(rightinputdir, vid, '00000' + ext) + '|'     # 5
                groups += os.path.join(rightinputdir, vid, '00000' + ext) + '|'     # 6
                groups += os.path.join(rightinputdir, vid, '00001' + ext) + '|'     # 7
                groups += os.path.join(rightinputdir, vid, '00002' + ext) + '|'     # 8
                groups += os.path.join(rightinputdir, vid, '00003' + ext) + '|'     # 9

                groups += os.path.join(lefttargetdir, vid, '00000' + ext) + '|'
                groups += os.path.join(lefttargetdir, vid, '00001' + ext) + '|'
                groups += os.path.join(lefttargetdir, vid, '00002' + ext) + '|'

                groups += os.path.join(righttargetdir, vid, '00000' + ext) + '|'
                groups += os.path.join(righttargetdir, vid, '00001' + ext) + '|'
                groups += os.path.join(righttargetdir, vid, '00002' + ext)

            elif idx == 1:
                groups += os.path.join(leftinputdir, vid, '00000' + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '00001' + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '00002' + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '00003' + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '00004' + ext) + '|'

                groups += os.path.join(rightinputdir, vid, '00000' + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '00001' + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '00002' + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '00003' + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '00004' + ext) + '|'

                groups += os.path.join(lefttargetdir, vid, '00001' + ext) + '|'
                groups += os.path.join(lefttargetdir, vid, '00002' + ext) + '|'
                groups += os.path.join(lefttargetdir, vid, '00003' + ext) + '|'

                groups += os.path.join(righttargetdir, vid, '00001' + ext) + '|'
                groups += os.path.join(righttargetdir, vid, '00002' + ext) + '|'
                groups += os.path.join(righttargetdir, vid, '00003' + ext)

            elif idx == (len(frames) - 2):
                groups += os.path.join(leftinputdir, vid, '{:05d}'.format(len(frames) - 5) + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '{:05d}'.format(len(frames) - 4) + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '{:05d}'.format(len(frames) - 3) + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '{:05d}'.format(len(frames) - 2) + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '{:05d}'.format(len(frames) - 1) + ext) + '|'

                groups += os.path.join(rightinputdir, vid, '{:05d}'.format(len(frames) - 5) + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '{:05d}'.format(len(frames) - 4) + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '{:05d}'.format(len(frames) - 3) + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '{:05d}'.format(len(frames) - 2) + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '{:05d}'.format(len(frames) - 1) + ext) + '|'

                groups += os.path.join(lefttargetdir, vid, '{:05d}'.format(len(frames) - 4) + ext) + '|'
                groups += os.path.join(lefttargetdir, vid, '{:05d}'.format(len(frames) - 3) + ext) + '|'
                groups += os.path.join(lefttargetdir, vid, '{:05d}'.format(len(frames) - 2) + ext) + '|'

                groups += os.path.join(righttargetdir, vid, '{:05d}'.format(len(frames) - 4) + ext) + '|'
                groups += os.path.join(righttargetdir, vid, '{:05d}'.format(len(frames) - 3) + ext) + '|'
                groups += os.path.join(righttargetdir, vid, '{:05d}'.format(len(frames) - 2) + ext)

            elif idx == (len(frames)-1):
                groups += os.path.join(leftinputdir, vid, '{:05d}'.format(len(frames)-4) + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '{:05d}'.format(len(frames)-3) + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '{:05d}'.format(len(frames)-2) + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '{:05d}'.format(len(frames)-1) + ext) + '|'
                groups += os.path.join(leftinputdir, vid, '{:05d}'.format(len(frames)-1) + ext) + '|'

                groups += os.path.join(rightinputdir, vid, '{:05d}'.format(len(frames)-4) + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '{:05d}'.format(len(frames)-3) + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '{:05d}'.format(len(frames)-2) + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '{:05d}'.format(len(frames)-1) + ext) + '|'
                groups += os.path.join(rightinputdir, vid, '{:05d}'.format(len(frames)-1) + ext) + '|'

                groups += os.path.join(lefttargetdir, vid, '{:05d}'.format(len(frames)-3) + ext) + '|'
                groups += os.path.join(lefttargetdir, vid, '{:05d}'.format(len(frames)-2) + ext) + '|'
                groups += os.path.join(lefttargetdir, vid, '{:05d}'.format(len(frames)-1) + ext) + '|'

                groups += os.path.join(righttargetdir, vid, '{:05d}'.format(len(frames)-3) + ext) + '|'
                groups += os.path.join(righttargetdir, vid, '{:05d}'.format(len(frames)-2) + ext) + '|'
                groups += os.path.join(righttargetdir, vid, '{:05d}'.format(len(frames)-1) + ext)



            else:
                for i in range(idx-2, idx+3):
                    groups += os.path.join(leftinputdir, vid, '{:05d}'.format(i) + ext) + '|'
                for i in range(idx-2, idx+3):
                    groups += os.path.join(rightinputdir, vid, '{:05d}'.format(i) + ext) + '|'

                for i in range(idx - 1, idx + 2):
                    groups += os.path.join(lefttargetdir, vid, '{:05d}'.format(i) + ext) + '|'
                for i in range(idx - 1, idx + 1):
                    groups += os.path.join(righttargetdir, vid, '{:05d}'.format(i) + ext) + '|'
                groups += os.path.join(righttargetdir, vid, '{:05d}'.format(idx + 1) + ext)

            with open(os.path.join(outputdir, 'traindata_sceneflow_twotarget_warploss.txt'), 'a') as f:
                f.write(groups + '\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--leftinput', type=str, default='./SceneFlow_video_train/lr_x4/input_left', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--righttinput', type=str, default='./SceneFlow_video_train/lr_x4/input_right', metavar='PATH', help='root dir to save low resolution images')

    parser.add_argument('--target_left', type=str, default='./SceneFlow_video_train/target/target_left', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--target_right', type=str, default='./SceneFlow_video_train/target/target_right', metavar='PATH', help='root dir to save high resolution images')

    parser.add_argument('--output', type=str, default='./SceneFlow_video_train/', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    leftinputdir = args.leftinput
    rightinputdir = args.righttinput

    lefttargetdir = args.target_left
    righttargetdir = args.target_right

    outputdir = args.output
    ext = args.ext

    main()