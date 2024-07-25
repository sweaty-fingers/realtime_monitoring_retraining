import argparse
import os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

import shutil
import threading
from save_init_model import TEST_DIR as model_dir

SAVED_MODEL_DIR = model_dir
VOC_DATA_DIR = os.path.join(ROOT_DIR, "retraining" , "data", "data", "VOCdevkit")

save_model_filename = "save_init_model.py"
voc_data_filename = "voc_dataset.py"

CAPTURED_IMAGE_DIR = os.path.join(ROOT_DIR, 'captured_images')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

def delete_folder_contents(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Delete the entire folder and its contents
        shutil.rmtree(folder_path)
        print(f"Deleted the folder and its contents: {folder_path}")
    else:
        print(f"The folder does not exist: {folder_path}")


def run_script(script_name):
    os.system(f"python {script_name}")


def main():
    parser = argparse.ArgumentParser(description='Example of argparse with boolean flag')

    parser.add_argument('--delete_logs', action='store_true', help='Set flag to True')
    parser.add_argument('--video', action='store_true', help='Set flag to True')
    # parser.add_argument('--no-flag', action='store_false', dest='flag', help='Set flag to False')
    
    args = parser.parse_args()

    if not os.path.exists(SAVED_MODEL_DIR):
        print(f"Load saved model, dir {SAVED_MODEL_DIR}")
        os.system(f"python {os.path.join(ROOT_DIR, save_model_filename)}")
    
    if not os.path.exists(VOC_DATA_DIR):
        print(f"Load voc data, dir {VOC_DATA_DIR}")
        os.system(f"python {os.path.join(os.path.dirname(os.path.dirname(VOC_DATA_DIR)), voc_data_filename)}")

    if args.delete_logs:
        print(f"Delete previous logs")
        delete_folder_contents(CAPTURED_IMAGE_DIR)
        delete_folder_contents(LOG_DIR)
    

    threads = []
    if args.video:
        # os.system(f"python {os.path.join(ROOT_DIR, "running_code", "device_simulation.py")}")
        thread = threading.Thread(target=run_script, args=(os.path.join(ROOT_DIR, "running_code", "device_simulation.py"),))
    else:
        thread = threading.Thread(target=run_script, args=(os.path.join(ROOT_DIR, "running_code", "just_make_realtime_image.py"),))
        # os.system(f"python {os.path.join(ROOT_DIR, "running_code", "just_make_realtime_image.py")}")
    
    thread.start()
    threads.append(thread)

    thread = threading.Thread(target=run_script, args=(os.path.join(ROOT_DIR, "running_code", "make_log.py"),))
    thread.start()
    threads.append(thread)

    thread = threading.Thread(target=run_script, args=(os.system(f"streamlit run {os.path.join(ROOT_DIR, "dashboard", "app.py")}"),))
    thread.start()
    threads.append(thread)
    

    # 모든 스레드가 종료될 때까지 대기
    for thread in threads:
        thread.join()
    

    

if __name__ == "__main__":
    main()