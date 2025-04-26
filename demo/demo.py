import os
import sys
import argparse
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.evaluation.tracker import Tracker
from lib.train.admin import create_default_local_file_ITP_train
from lib.test.evaluation import create_default_local_file_ITP_test

def initEnv():
    workspace_dir = os.path.realpath('.')
    data_dir = os.path.realpath('data')
    save_dir = os.path.realpath('.')
    create_default_local_file_ITP_train(workspace_dir, data_dir)
    create_default_local_file_ITP_test(workspace_dir, data_dir, save_dir)

def main():
    parser = argparse.ArgumentParser(description="视频目标跟踪演示程序")
    parser.add_argument("-i", "--video_path", type=str, required=True, help="输入视频路径")
    parser.add_argument("--tracker_name", type=str, default="mcitrack", help="跟踪器名称")
    parser.add_argument("--tracker_param", type=str, default="mcitrack_t224", help="跟踪器参数文件名称")
    parser.add_argument("--debug", type=int, default=0, help="调试级别")

    args = parser.parse_args()

    initEnv()

    tracker = Tracker(args.tracker_name, args.tracker_param, dataset_name=None)

    tracker.run_video(args.video_path, debug=args.debug, save_results=True)


if __name__ == "__main__":
    main()