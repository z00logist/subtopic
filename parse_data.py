import os
import re
import sys

import pandas as pd
from pysubparser import parser  # https://github.com/fedecalendino/pysub-parser
from pysubparser.cleaners import ascii, brackets, formatting, lower_case
from typing import Generator

SUB_EXTENSIONS = ['.ass', '.ssa', '.srt', '.sub', '.txt']


class UnicodeErrorUTF16(UnicodeError):
    def __init__(self):
        super().__init__('The file is encoded in UTF-16')


def read_txt(path: str) -> str:
    with open(path, mode='r') as file:
        subtitles = file.readlines()
        
    subtitles = [subtitle.rstrip() 
                 for subtitle in subtitles 
                 if subtitle.rstrip()]  # Иначе могут остаться пустые строки после удаления
    subtitles = ' '.join(subtitles)

    return subtitles
    
    
def read_srt(path: str) -> str:
    def preprocessing(subtitles: Generator) -> str:
        subtitles = formatting.clean(
            ascii.clean(brackets.clean(lower_case.clean(subtitles)))
        )
        subtitles = [clean_string_num(str(subtitle)) for subtitle in subtitles]
        subtitles = ' '.join(subtitles)
        if not subtitles:
            raise UnicodeErrorUTF16
        
        return subtitles
    
    def clean_string_num(original: str) -> str:
        cleaned = re.sub(r'\d+ > ', '', original)
        return cleaned

    try:
        subtitles = parser.parse(path, encoding='cp1251')
        subtitles = preprocessing(subtitles)
    except UnicodeErrorUTF16:
        subtitles = parser.parse(path, encoding='utf-16')
        subtitles = preprocessing(subtitles)

    return subtitles


def read_and_format_file(path: str) -> str:
    file_extension = os.path.splitext(path)[1]
    if file_extension == '.txt':
        subtitles = read_txt(path)
    elif file_extension == '.srt':
        subtitles = read_srt(path)
    return subtitles


def parse_data(path: str) -> pd.DataFrame:
    raw_data = []
    total_sub_files = 0

    for dir_path, _, filenames in os.walk(path):
        if filenames:
            filenames = list(
                filter(lambda f: os.path.splitext(f)[1] in SUB_EXTENSIONS, filenames)
            )
            
            for filename in filenames:
                total_sub_files += 1
                full_path = dir_path + '/' + filename
                
                try:    
                    raw_text = read_and_format_file(path=full_path)
                    print(f'[+]\t{full_path: <50}\tPARSED')
                except Exception as ex:
                    print(f'[-]\t{full_path: <50}\tERROR')
                
                single_data = {
                    'author': dir_path.split('/')[-1].title(),
                    'name': os.path.splitext(filename)[0],
                    'text': raw_text,
                }
                raw_data.append(single_data)

    texts_df = pd.DataFrame(data=raw_data)

    return texts_df, total_sub_files


def main():
    DATA_PATH = f'./{sys.argv[1]}'
    PARSED_DATA_PATH = DATA_PATH + '/' + sys.argv[2]
    
    try:
        texts_df, total_sub_files = parse_data(DATA_PATH)
        texts_df.to_csv(PARSED_DATA_PATH, index=False)
    except Exception as ex:
        print(f'An error occured while parsing the data:\n{ex}\n')
    finally:
        print(f'\nParsed {len(texts_df)}/{total_sub_files} items into {PARSED_DATA_PATH}\n')


if __name__ == '__main__':
    main()
