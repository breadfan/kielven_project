import fitz
import re
import pandas as pd
import numpy as np

from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt


PDF_PATH = 'data//pdf_in_karjalan//russko-karelskij_slovar.pdf'

def form_words(blocks):
    words, words_wo_endings = list(), list()    
    for b in blocks:  # iterate through the text blocks
        for l in b["lines"]:  # iterate through the text lines
            w = ''
            for s in l["spans"]:  # iterate through the text spans
                if s['text'] == '|': # бывают разных флагов поэтому сначала это условие
                    w += '|'
                elif s["flags"] == 20:  # 20 номер — жирный шрифт, таргетные слова
                    word = s['text']
                    word = word.replace("-\n", " ")
                    word = word.replace("\n", " ")
                    word = word.replace(u'\xa0', u' ')
                    word = word.replace(u'\xad', '')
                    word = word.replace('\uf009', '')
                    word = re.sub(r'[.]', '', word)
                    word = word.strip()
                    if word and word[-1].isdigit(): # если цифры в конце
                        if w:
                            bar_ind = w.find('|')
                            words.append(w)
                            if bar_ind != -1:
                                words_wo_endings.append(w[:bar_ind])
                            else:
                                words_wo_endings.append(w)
                        elif word[:-2]:
                            words.append(word[:-2]) # до цифры один
                            bar_ind = word.find('|')
                            if bar_ind != -1:
                                words_wo_endings.append(word[:bar_ind])
                            else:
                                words_wo_endings.append(w[:bar_ind])
                        w = word[-1]
                    else:
                        w += word
            if w:
                words.append(w)
                bar_ind = w.find('|')
                if bar_ind != -1:
                    words_wo_endings.append(w[:bar_ind])
                else:
                    words_wo_endings.append(w)
    return words, words_wo_endings

def gather_karelian_words(block, rus_word):
    """
    Gather all karelian words from the blocks
    :param block_split: list of blocks
    :return: list of karelian words
    """
    kar_to_rus = dict()
    pattern = r'\);|-;' 
    block_split = re.split(pattern, block)
    if len(block_split) == 1 and ';' in block_split[0]:
        block_split = re.split(r';', block_split[0])
    karelian_words = []
    for line in block_split:
        if '~' in line or line.find('-') == 0 or line.find('-') == 1: # если тильда — то это словосочетания и уже не нужны; если неправильно разделилось, то в начале дефис
            continue
        if line.strip():
            to_add = ''
            if '(' in line:
                if line.find('(') == 0:
                    line = line[line.find(')')+1:]
                br_left, br_right = line.find('('), line.find(')')
                to_add = line[:br_left]
            else:
                to_add = line
            to_add = to_add.strip(' .,;:').replace('|', '')
            karelian_words.append(to_add)
    for word in karelian_words:
        kar_to_rus[word] = rus_word
    return kar_to_rus


def get_word_span(text, words):
    word_span = list()
    second = 0
    for i in range(len(words)):
        first = text[second:].find(words[i])
        first += second
        if i!=len(words)-1:
            second = text[first:].find(words[i+1])
        else:
            second = len(text)
        second += first
        word_span.append((words[i].strip(), text[first+len(words[i]):second].strip()))
    return word_span

def mod_words(word_span, words_wo_endings, vars_meanings):
    new_words, new_words_wo_ending = list(), list()
    new_word_span = list()
    full_words = list()
    for num, (word, span) in enumerate(word_span):
        if word.isupper(): # если это одна заглавная буква (сверху страницы)
            continue
        if span:
            if word.isdigit() or (len(word)==9 and word.find(' — ') != -1):
                if span.find('(') < 10:
                    br_left = span.find('(')
                    br_right = span.find(')')
                    meaning = span[br_left:br_right+1]
                    new_word = vars_meanings[0] + ' ' + meaning
                    new_span = span[br_right+1:].strip()
                else:
                    new_word = vars_meanings[0]
                    new_span = span
                new_word_span.append((new_word.strip('.: '), new_span))
                new_words.append(new_word.strip('.: '))
                new_words_wo_ending.append(vars_meanings[1])
                full_words.append(vars_meanings[0].replace('|', ''))
            else:
                new_word_span.append((word, span))
                new_words.append(word.strip('.: '))
                new_words_wo_ending.append(words_wo_endings[num].strip('.: '))
                full_words.append(word.replace('|', ''))
                vars_meanings[0] = word.strip('.: ')
                vars_meanings[1] = words_wo_endings[num].strip('.: ')
        if not word.isdigit() and not (len(word)==9 and word.find(' — ') != -1):
            vars_meanings[0] = word
            vars_meanings[1] = words_wo_endings[num]
            
    assert len(new_words_wo_ending) == len(new_words), 'длины не совпадают'
    word_to_lexeme = dict(zip(new_words, new_words_wo_ending))
    word_to_fullword = dict(zip(new_words, full_words))
    return new_word_span, word_to_lexeme, word_to_fullword, \
        new_words, new_words_wo_ending

def add_related(kar_part, rus_part):
    br_left, br_right = kar_part.find('('), kar_part.find(')')
    replacement = re.split(r'[(,)]', kar_part[br_left:br_right+1])
    replacement = [r.strip() for r in replacement if r.strip()]
    clean_part = kar_part[:br_left-1] + kar_part[br_right+1:]
    brackets_part = kar_part[br_left+1:br_right]
    
    add_kar, add_rus = list(), list()
    if br_right == len(kar_part)-1:
        if ',' in brackets_part:
            wp_ind = kar_part[:br_left-1].rfind(' ')
            for repl in replacement:
                add_kar.append(kar_part[:wp_ind+1] + repl + kar_part[br_right+1:])
                add_rus.append(rus_part)
        else:
            if 1 < len(brackets_part.split()):
                add_kar.append(replacement[0])
                add_rus.append(rus_part)
            else:
                wp_ind = kar_part[:br_left-1].rfind(' ')
                add_kar.append(kar_part[:wp_ind+1] + replacement[0])
                add_rus.append(rus_part)
    else:
        wp_ind = kar_part[:br_left-1].rfind(' ')
        for repl in replacement:
            add_kar.append(kar_part[:wp_ind+1] + repl + kar_part[br_right+1:])
            add_rus.append(rus_part)
    return add_kar, add_rus, clean_part


def parse_one_pair(word, example, word_to_lexeme, word_to_fullword):
    corpus_kar, corpus_rus = list(), list()
    add_kar, add_rus = list(), list()
    example = example.replace('  ', ' ')
    tilda_ind = example.find('~')
    to_repl = word_to_lexeme[word]
    if tilda_ind != -1:
        if tilda_ind == len(example)-1 or example[tilda_ind+1] == ' ':
            to_repl = word_to_fullword[word]
    
    s = example.replace('~', to_repl).strip()       
    
    match = re.search(r'[а-яА-Я](?!.*[а-яА-Я])', s)
    eng_match = re.search(r'[a-zA-Z]', s)
    if eng_match is not None and eng_match.start() < match.end():
        split_index = eng_match.start()
    else:
        split_index = match.end()
    
    if s[0] != '(' and s[split_index] != ')':
        rus_part = s[:split_index]
        kar_part = s[split_index+1:]
        corpus_rus.append(rus_part)
        # сегмент где исключаем однокоренные
        if '(' in kar_part:
            add_kar, add_rus, clean_part = add_related(kar_part, rus_part)
            corpus_kar.append(clean_part)
        else:
            corpus_kar.append(kar_part)
    return corpus_kar, corpus_rus, add_kar, add_rus

def parse_sentences(new_word_span, 
                    word_to_lexeme, 
                    word_to_fullword, 
                    ):
    kar_to_rus = dict()
    kar_corpus, rus_corpus = list(), list()
    kar_add, rus_add = list(), list()
    pattern = r'[;◊]'
    for word, span in new_word_span:
        block_split = re.split(pattern, span)
        k2r = gather_karelian_words(span, word)
        kar_to_rus.update(k2r)
        for example in block_split[1:]:
            try:
                kar_corp, rus_corp, \
                    kara, rusa = parse_one_pair(word, example, \
                        word_to_lexeme, word_to_fullword)
                kar_corpus.extend(kar_corp)
                rus_corpus.extend(rus_corp)
                kar_add.extend(kara)
                rus_add.extend(rusa)
            except:
                continue
    return kar_corpus, rus_corpus, kar_add, rus_add, kar_to_rus

def parse_one_page(page, vars_meanings):
    text = page.get_text("text")
    text = text.replace(u'\xa0', u' ').replace(u'\xad', '')
    text = text.replace("-\n", "").replace("\n", "")
    text = text.replace('  ', ' ')
    text = text.replace('\uf009', '')
    
    # забираем слова и слова без окончаний 
    blocks = page.get_text("dict", flags=11)["blocks"]
    words, words_wo_endings = form_words(blocks)
    word_span = get_word_span(text, words)
    new_word_span, word_to_lexeme, word_to_fullword, \
        new_words, new_words_wo_ending = mod_words(word_span, words_wo_endings, vars_meanings)
    kar_corpus, rus_corpus, kar_add, rus_add, k2r = parse_sentences(new_word_span, word_to_lexeme, 
                                                               word_to_fullword)
    return kar_corpus, rus_corpus, kar_add, rus_add, word_to_fullword, k2r

def parse_suuri_vena():
    kar_to_rus = dict()
    
    troubles = 0
    kar_corpus, rus_corpus = list(), list()
    kar_add, rus_add = list(), list()
    pdf_path = PDF_PATH
    doc = fitz.open(pdf_path)
    vars_meanings = [None, None]
    word_to_fullword = dict()
    for page_ind in tqdm(range(12,400)): # (12, 400)
        try:
            kar_ext, rus_ext, kara, rusa, wtf, k2r = parse_one_page(doc[page_ind], vars_meanings)
            kar_to_rus.update(k2r)
            kar_corpus.extend(kar_ext)
            rus_corpus.extend(rus_ext)
            kar_add.extend(kara)
            rus_add.extend(rusa)
            word_to_fullword.update(wtf)
        except:
            print(page_ind)
            troubles += 1 
            continue
        
    print(f"Troubles: {troubles}")
    return kar_corpus, rus_corpus, kar_add, rus_add, word_to_fullword, kar_to_rus

if __name__ == '__main__':
    kar_corpus, rus_corpus, kar_add, rus_add, wtf, kar_to_rus = parse_suuri_vena()
    
    # сохраняем 
    ctr = 0
    res_kar, res_rus = list(), list()
    for (s_kar, s_rus) in zip(kar_corpus, rus_corpus):
        if re.search(r'[а-яА-я]', s_kar) is not None:
            ctr+=1
        else:
            res_kar.append(s_kar)
            res_rus.append(s_rus)
    print(ctr)
    print(len(kar_add), len(rus_add))
    # и в синонимах
    for (s_kar, s_rus) in zip(kar_add, rus_add):
        if re.search(r'[а-яА-я]', s_kar) is not None:
            ctr+=1
        else:
            res_kar.append(s_kar)
            res_rus.append(s_rus)
    print(ctr)
    num = -100
    print(kar_add[num], rus_add[num])
    print(res_kar[num], res_rus[num])
    res_df = pd.DataFrame({'rus': res_rus, 'kar': res_kar})
    # res_df = res_df.sample(frac=1).reset_index(drop=True)
    res_df['split'] = 'train'
    res_df.to_csv('data//suuri_vena_clean2.csv', index=False)
    res_rus = rus_corpus.copy()
    res_kar = kar_corpus.copy()
    res_rus.extend(rus_add)
    res_kar.extend(kar_add)
    res_df = pd.DataFrame({'rus': res_rus, 'kar': res_kar})
    # res_df = res_df.sample(frac=1).reset_index(drop=True)
    res_df['split'] = 'train'
    res_df.drop_duplicates(ignore_index=True, inplace=True)
    res_df.to_csv('data//suuri_vena2.csv', index=False)
    # статистика по количеству слов
    lens = [len(row.split()) for row in kar_corpus]
    plt.hist(lens, bins=100)