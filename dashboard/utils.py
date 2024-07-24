import os
import streamlit as st

def list_files_with_extension(directory, extension):
    """
        # 사용 예
        directory_path =   # 대상 디렉토리 경로
        file_extension = png  # 찾고자 하는 파일 확장자

        files = list_files_with_extension(directory_path, file_extension)
        print(files)

        # 지정한 디렉토리 내의 파일 리스트를 가져옴
    """
    
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]

def iter_files_with_extension(directory, extension):
    """
        # 사용 예
        directory_path =   # 대상 디렉토리 경로
        file_extension = png  # 찾고자 하는 파일 확장자

        files = list_files_with_extension(directory_path, file_extension)
        print(files)

        # 지정한 디렉토리 내의 파일 리스트를 가져옴
    """
    for f in os.listdir(directory):
        if f.endswith(extension):
            yield os.path.join(directory, f)


    