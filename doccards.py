import fitz
from operator import itemgetter

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import string
import numpy as np
import os
from PIL import Image, ImageDraw
from tqdm import tqdm
import colorsys
from collections import Counter
import rpack
import panel as pn
from bokeh.models import TextInput, Tooltip
import math
from functools import partial
import matplotlib as mpl
import matplotlib.cm as cm
from pathlib import Path


def main():

    pdf_path = Path("input/").glob("*.pdf")
    pdf_files = [str(file.absolute()) for file in pdf_path]
    print("input files: ", pdf_files)

    docCards = []

    for i, path in enumerate(pdf_files):
        doc = fitz.open(path)
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        print("current file: ", stem)

        # for all documents, get the Document Card
        docCard = getDocumentCard(doc, stem)
        docCards.append(docCard)

        print("done ", i+1, "/", len(pdf_files))

    # display it in a FlexBox
    box = pn.FlexBox(*(docCards))
    box.show()


def getDocumentCard(doc, stem):

    pages_text = getPagesText(doc)

    maxKeyterms = 9


    ### TEXT AND IMAGE PREPROCESSING ###

# 1. Text Extraction

    # first find different sections in the document
    # identify headlines, authors and abstract

    blocks_list = getBlocks(doc)
    #font_sizes_csorted, font_sizes_fsorted = getFontSizes(doc) # not used in finished code
    title, authors = getTitleAuthors(doc)
    abstract = getAbstract(blocks_list)
    
    # get pdf pages as images
    getPageImages(doc, stem)

    # 1.1 Preprocessing

        # Sentence Splitting, Part-of-Speech Tagging, Noun Phrase Chunking
        # Base Form Reduction (Lemmatizing)
    
    blocks_sen_list = sentenceSplitting(blocks_list)
    blocks_words_pos_list = partOfSpeechTagging(blocks_sen_list)
    blocks_chunked_sen_list, blocks_chunked_grammar_list, blocks_chunked_words_list = nounPhraseChunking(blocks_words_pos_list)
    blocks_lemma_words_list = lemmatizing(blocks_chunked_words_list)

    # 1.2 Candidate Filtering

        # eliminate stopwords, if there are some left after preprocessing
        # noise like punctuation and verbs are already removed in the previous Preprocessing
    blocks_filtered_words = eliminateStopwords(blocks_lemma_words_list)

    # 1.3 Special Noun-Phrase Processing

        # build every rightmost sub phrases (generalizations) of compound nouns and also consider them
    words = getSpecialWords(blocks_filtered_words)

# 2. Key Term Extraction
    
    # 2.1 Term Scoring

        # build vector for each term, e.g. "term" = [0,1,0,0,2,0] (# occurrences of "term" in every block)
    terms_list_filtered, terms_vector_filtered, terms_summed_filtered = buildVectors(words)

        # boost compund nouns by doubling their occurence counts
    terms_vector_filtered, terms_summed_filtered = doubleOccurence(terms_list_filtered, terms_vector_filtered, terms_summed_filtered)
    terms_scores_norm, terms_and_scores = calcDerivation(words, terms_vector_filtered, terms_summed_filtered, terms_list_filtered)
   
    # 2.2 Top-k Terms

        # top-k terms with highest score are extracted
        # called now for max and again when k is defined
    top_k_terms, top_k_terms_scores = find_top_k_terms(maxKeyterms, terms_and_scores)

# 3. Image Extraction
    
        # extract all imgs and belonging captions
    blocks_data = extractImages(doc, stem)

# 4. Image Preprocessing
    
    # 4.1 Importance

        # important = key term found in caption
        # resize image according to its importance
    resized_data = resizeImages(blocks_data, top_k_terms_scores, stem)

    # 4.2 Classification

    img_classes = classifyImages(blocks_data, resized_data, stem)


    ### VISUALIZATION ###

# 5. Image Preparation
    
    # 5.1 Image Selection
    
        # 2 constraints: 

            # 1. 1 img from each class
            #   if one class not represented -> discard last img and take largest img of missing class

            # 2. area of all imgs should be larger than 25% of largest img,
            #   if not -> discard
    filtered_imgs = selectImages(img_classes)

    # 5.2 Image Positioning

        # with img packing tool
    img_positions, img_positions_extended = getImagePositions(filtered_imgs)

        # when positioning the images, calculate the free space, where the keywords are inserted
    max_x_doccard, max_y_doccard = getDocCardWidthHeight(img_positions_extended)
    free_space_rectangles = getFreeSpaceRect(img_positions_extended, max_x_doccard, max_y_doccard)
    sorted_rectangles = sortRectangles(free_space_rectangles)

    # 5.3 Img Buttons

        # create the overview pane
    #overview = makeOverview(max_x_doccard, max_y_doccard, filtered_imgs, free_space_rectangles, background_col, img_positions)
        # get tooltips and page number for the images
    img_pages, tooltips, abstact_tooltip = getTooltipsPageNumbs(filtered_imgs, resized_data, abstract)
    img_buttons, img_panes = makeImgButtons(filtered_imgs, tooltips, img_positions, stem)

# 6. Term Preparation

    # 6.1 Term Selection

        # get k most important key terms
    top_k_terms_docCard, top_k_terms_scores_docCard = find_top_k_terms_docCard(maxKeyterms, max_x_doccard, max_y_doccard, filtered_imgs, terms_and_scores)
        
    # 6.2 Term Positioning and Buttons
        
        # position key terms
    term_sizes_positions, term_panes, str_buttons = posTerms(top_k_terms_scores_docCard, sorted_rectangles)
        
        # count how often a term appears on each page
    if term_sizes_positions:
        terms_amount_norm, terms_amount, min_amount, max_amount = countTermAmount(pages_text, term_sizes_positions)
        terms_colors = getTermColors(terms_amount_norm, terms_amount, min_amount, max_amount)
    else:
        terms_colors = [(0,0,0)]

# 7. Document Card
    
    # 7.1 Background Color

        # get most frequent H value in HSV images for background except grey colors
    if filtered_imgs:
        background_col, background_col_str = getBackgroundColor(filtered_imgs, stem)
    else:   
        background_col = [240,240,240]
        background_col_str = "rgb(240,240,240)"

    # 7.2 Card Creation
    docCard = createDocCard(doc, authors, title, background_col, background_col_str, abstact_tooltip, max_x_doccard, max_y_doccard, free_space_rectangles, img_buttons, img_panes, term_panes, str_buttons, terms_colors, img_pages, stem)

    return docCard



def getPagesText(doc):
    pages_text = []
    for page in doc:
        page_text = page.get_text()
        pages_text.append(page_text)
    return pages_text

def getBlocks(doc):
    doc_blocks_list = []
    filtered_doc_blocks_list = []

    for page in doc:
        output = page.get_text("blocks")
        pre_block_id = 0
    
        # output = (x0, y0, x1, y1, "lines in the block", block_id, block_type)
    
        for block in output:
            if block[6] == 0:                   # if block type is text (not img)
                #if pre_block_id != block[5]:    # if previous block is another block than the current
                    
                #lines = block[4]
                doc_blocks_list.append(block)

    # filter out References and the pages after that
    for block in doc_blocks_list:
        if (block[4] == "References") or (block[4] == "References\n") or (block[4] == "REFERENCES") or (block[4] == "REFERENCES\n"):
            break;
        else:
            filtered_doc_blocks_list.append(block)
    
    #print("filtered doc blocks list: ", filtered_doc_blocks_list)
    return filtered_doc_blocks_list

def getFontSizes(doc):
    styles = {}
    font_counts = {}

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        identifier = "{0}".format(span['size'])
                        styles[identifier] = {'size': span['size'], 'font': span['font']}
                        font_counts[identifier] = font_counts.get(identifier, 0) + 1    # count fonts usage
                    
    font_counts_csorted = sorted(font_counts.items(), key = itemgetter(1), reverse = True)

    list_font_counts = []
    for item in font_counts_csorted:
        itemlist = list(item)
        string = itemlist[0]
        integer = float(string)
        itemlist[0] = integer
        list_font_counts.append(itemlist)
    
    font_counts_fsorted = sorted(list_font_counts, key = itemgetter(0))

    return font_counts_csorted, font_counts_fsorted

def getTitleAuthors(doc):
    # get title (take first 10 blocks and look for max font)
    # get authors (take first 10 blocks and look for second max)

    maxFontSize = 0
    maxFontSizeWords = []

    firstPage = doc[0] 
    blocks = firstPage.get_text("dict")["blocks"]
    for block in blocks[:10]:
        if block["type"] == 0:
            for line in block["lines"]:
                for span in line["spans"]:
                    if len(span['text']) > 1:
                        maxFontSizeWords.append((span['size'], span['text'], block['number']))

    sortedMaxWords = sorted(maxFontSizeWords, key=lambda span: -span[0])

    #print("sorted max words: ", sortedMaxWords)

    maxWords = sortedMaxWords[0]
    #print("maxWords: ", maxWords)

    for word in sortedMaxWords:
        if word[0] < maxWords[0]:
            secondMaxWords = word
            break;
    #print("secondMaxWords: ", secondMaxWords)

    titleBlock = []

    for sortedMaxWord in sortedMaxWords[::-1]:
        if maxWords[0] == sortedMaxWord[0]:
            titleBlock.append(sortedMaxWord)

    authorsBlock = []

    for sortedMaxWord in sortedMaxWords[::-1]:
        if secondMaxWords[0] == sortedMaxWord[0]:
            if authorsBlock and (authorsBlock[-1] != ","):
                authorsBlock.append((0, ",", 0))
            authorsBlock.append(sortedMaxWord)

    # get title text
    title = ""

    for span in titleBlock:
        title = '{} {}'.format(span[1], title)
    #print("title: ", title)

    # get authors text
    authors = ""

    for span in authorsBlock:
        authors = '{} {}'.format(span[1], authors)

    if authors[-1] == "," or authors[-1] == " ":
        authors = authors[:-1]
    #print("authors: ", authors)

    return title, authors

def getAbstract(blocks_list):
    text = ''
    for block in blocks_list:
        if ("Abstract" in block[4]) or ("Abstract\n" in block[4]) or ("ABSTRACT" in block[4]) or ("ABSTRACT\n" in block[4]): # has to be upper case (and the first word in the block) 
            text = block[4]
    abstract = text.replace("\n", "")
    return abstract

def getPageImages(doc, stem):
    for page in doc:
        pix = page.get_pixmap(dpi = 500)
        pix.save("extracted_pages/%s-page-%i.png" % (stem, page.number+1))


def sentenceSplitting(blocks_list):
    blocks_sen_list = []

    for block in blocks_list:
        block_string = block[4]
        block_sen = sent_tokenize(block_string)
        blocks_sen_list.append(block_sen)

    return blocks_sen_list # every element in this list is a block, which contains its splitted sentences

def partOfSpeechTagging(blocks_sen_list):
    blocks_words_list = []
    blocks_words_pos_list = []

    for block in blocks_sen_list:
        words_list = []
        words_pos_list = []
        for sentence in block:
            words_in_sentence = word_tokenize(sentence)
            words_pos_tags = nltk.pos_tag(words_in_sentence)
            words_list.append(words_in_sentence)
            words_pos_list.append(words_pos_tags)
        blocks_words_list.append(words_list)
        blocks_words_pos_list.append(words_pos_list)
    
    return blocks_words_pos_list

def nounPhraseChunking(blocks_words_pos_list):
    # Die Grammatik kann noch erweitert werden bzw. angepasst werden. 
    # Momentan chunke ich noch keine Nomen, die am Satzanfang stehen mit darauffolgenden Nomen, 
    # da diese als NNP angezeigt werden, obwohl sie NN sind.

    grammar = """
        NP: {<VBG>*<JJ>*<NN>+<NNS>*<NNP>*}
            {<VBG>*<JJ>*<NNS>+<NN>*<NNP>*}
            {<VBG>*<JJ>*<NNP>+<NN>*<NNS>*}
    """

    chunk_parser = nltk.RegexpParser(grammar)

    blocks_chunked_sen_list = []
    blocks_chunked_grammar_list = []

    for block in blocks_words_pos_list:
        chunked_sen_list = []
        chunked_grammar_list = []
        for sentence in block:
            chunked_sen_tree = chunk_parser.parse(sentence)
            chunked_sen_list.append(chunked_sen_tree)
        
            # find the subtrees with the correct grammar we are searching for
            # eliminate noise like punctuation and verbs
            for subtree in chunked_sen_tree.subtrees():
                if subtree.label() == 'NP': 
                    chunked_grammar_list.append(subtree)
                
                
        blocks_chunked_sen_list.append(chunked_sen_list)
        blocks_chunked_grammar_list.append(chunked_grammar_list)


    # filter out the words without its tag per sentence and block

    blocks_chunked_words_list = []

    for block in blocks_chunked_grammar_list:
        sen_words_list = []
        for sen_tree in block:
            words_list = []
            for word in sen_tree:
                # außerdem werden manche Satzzeichen als Adjektive gesehen, deswegen nochmal extra rausfiltern
                words_list.append(word[0])
            sen_words_list.append(words_list)
        blocks_chunked_words_list.append(sen_words_list)

        
    return blocks_chunked_sen_list, blocks_chunked_grammar_list, blocks_chunked_words_list

def lemmatizing(blocks_chunked_words_list):
    lemmatizer = WordNetLemmatizer()

    blocks_lemma_words_list = []

    for block in blocks_chunked_words_list:
        lemma_sen_list = []
        for sentence in block:
            lemma_words_list = []
            for word in sentence:
                lemma_word = lemmatizer.lemmatize(word)
                lemma_words_list.append(lemma_word)
            lemma_sen_list.append(lemma_words_list)
        blocks_lemma_words_list.append(lemma_sen_list)
    
    return blocks_lemma_words_list


def eliminateStopwords(blocks_lemma_words_list):

    stopwords = nltk.corpus.stopwords.words('english')
    extendStopwordsList = ['et', 'al', 'pp', 'e.g.', 'fig.', 'fig']
    stopwords.extend(extendStopwordsList)

    punctuations = string.punctuation + "“”’•"
    punct_without_hyphen = punctuations.replace("-", "")
    #print("Punctuations: ", punct_without_hyphen)

    blocks_filtered_words = []

    for block in blocks_lemma_words_list:
        sen_filtered_words = []
        for sentence in block:
            sentence_filtered = [word for word in sentence if len(word) >= 2]
            filtered_words = []
            for word in sentence_filtered:

                usable = True

                for punct in punct_without_hyphen:
                    if punct in word:
                        usable = False
                        
                if (word not in punct_without_hyphen) and usable:
                    lowercaseWord = word.casefold()         # casefold to ignore upper case letters
                    if (lowercaseWord not in stopwords):
                        if word:    # be sure that all arrays are not empty, otherwise there can be empty arrays in filtered words
                            filtered_words.append(word)
            
            if filtered_words:
                sen_filtered_words.append(filtered_words)
        if sen_filtered_words:
            blocks_filtered_words.append(sen_filtered_words)

    #print("blocks_filtered_words: ", blocks_filtered_words)
    return blocks_filtered_words

def getSpecialWords(blocks_filtered_words):
    blocks_special_words = []

    for block in blocks_filtered_words:
        sen_special = []
        for sentence in block:
            length = len(sentence)
            sen_special.append(sentence)
            new_sen = sentence.copy()
            for i in range(length-1):
                new_sen = new_sen.copy()
                new_sen.pop(0)
                sen_special.append(new_sen)
        blocks_special_words.append(sen_special)
            
    return blocks_special_words

def buildVectors(words):

    # all terms in the doc in one list
    terms_list = []
    for block in words:
        for term in block:
            if term not in terms_list:
                terms_list.append(term)
    
    # build empty vectors for each term in the list
    terms_vector = []

    for term in terms_list:
        term_vector = [0]*len(words) # for every term, there is now a vector with the length of #blocks
        terms_vector.append(term_vector)

    # fill vectors
    term_id = 0

    for term in terms_list:
        block_id = 0
        for block in words:
            counter = block.count(term)          # zähle occurrences of term in this block
            #print(term, counter)
            terms_vector[term_id][block_id] = counter
            block_id += 1         
        term_id += 1

    # sum up occurrences for all terms in whole doc
    terms_summed = []

    for term in terms_vector:
        sum = 0
        for occurence in term:
            sum += occurence
        terms_summed.append(sum)

    # only keep terms that occur at least 7 times
    terms_list_filtered = []
    terms_vector_filtered = []
    terms_summed_filtered = []

    index = 0

    for term_sum in terms_summed:
        if term_sum >= 7:
            terms_list_filtered.append(terms_list[index])
            terms_vector_filtered.append(terms_vector[index])
            terms_summed_filtered.append(terms_summed[index])
        index += 1

    #print("terms list filtered: ", terms_list_filtered)
    #print("terms summed filtered: ", terms_summed_filtered)
    return terms_list_filtered, terms_vector_filtered, terms_summed_filtered

def doubleOccurence(terms_list_filtered, terms_vector_filtered, terms_summed_filtered):
    index = 0
    for term in terms_list_filtered:
        if len(term)>=2:
            terms_vector_filtered[index] = terms_vector_filtered[index]*2
            terms_summed_filtered[index] = terms_summed_filtered[index]*2
        index += 1
    return terms_vector_filtered, terms_summed_filtered

def calcDerivation(words, terms_vector_filtered, terms_summed_filtered, terms_list_filtered):
    size_D = 0
    for block in words:
        size_D += len(block)

    terms_scores = []
    term_id = 0

    for term_vector in terms_vector_filtered:
        chi = 0
        block_id = 0
        for block in words:
            freq_t_s = term_vector[block_id]
            if freq_t_s > 0:
                freq_t_D = terms_summed_filtered[term_id]
                size_s = len(block)
                exp_freq = freq_t_D * (size_s/size_D)
                subtraction = freq_t_s - exp_freq
                upper_term = pow(subtraction, 2)
                sum = upper_term/freq_t_s
            else: sum = 0
            chi += sum
            block_id += 1
        terms_scores.append(chi)
        term_id += 1


    # normalize between 0 and 1
    terms_scores_norm = []

    min_score = min(terms_scores)
    max_score = max(terms_scores)

    for score in terms_scores:
        score_norm = (score - min_score) / (max_score - min_score)
        terms_scores_norm.append(score_norm)


    # concatenate terms with their score
    zipped_terms_and_scores = zip(terms_list_filtered, terms_scores_norm)
    terms_and_scores = list(zipped_terms_and_scores)
    
    
    return terms_scores_norm, terms_and_scores

def find_top_k_terms(k, terms_and_scores):

    #print("input: ", terms_and_scores)

    terms_and_scores = sorted(terms_and_scores, key=lambda x: x[1], reverse=True)
    #print("after sorting: ", terms_and_scores)
    top_k_terms_scores = terms_and_scores[:k]


    # keep more specific compound nouns

    output = top_k_terms_scores.copy()

    for term in top_k_terms_scores:
        #print("aktueller term: ", term[0])
        top_k_terms_scores_copy = top_k_terms_scores.copy() # copy the list without term
        top_k_terms_scores_copy.remove(term)
        term_length = len(term[0])
        #print("term lenght: ", term_length)
        for term_copy in top_k_terms_scores_copy: # wenn ein anderer term gefunden wird, der term enthält -> term wird aus der richtigen liste removed
            #print("vergleichen mit: ", term_copy[0])
            #print("term lenght: ", term_length)
            if len(term[0]) <= len(term_copy[0]):
                #print("ist der term das gleiche was hinten steht?: ", term_copy[0][-term_length:])
                if (term_copy[0][-term_length:] == term[0]):  # nach dem term soll nichts mehr kommen -> ist das letzte wort
                    #print("it is included in another term!")
                    output.remove(term)
                    terms_and_scores.remove(term)
                    break

    #print("output: ", output)

    top_k_terms = []
    for terms in output:
        top_k_terms.append(terms[0])

    return top_k_terms, output
    

def extractImages(doc, stem):

    workdir = 'extracted_images'
    blocks_data = []
    pixmaps = []

    imgnr = 1

    for page in doc:

        d = page.get_text("dict")
        blocks = d["blocks"]
        splitted_img_bboxes =[]
        block_number = 0
        img_found = False
    
        for block in blocks:
            block_number += 1
            if block["type"] == 1:         # img
                splitted_img_bboxes.append(block['bbox'])
                img_found = True
            elif block["type"] == 0:       # txt
                if (img_found == True):    # wenn ein img gefunden wurde, suche nach der description
                    # 1. schauen ob text mit Fig / Tab beginnt und den block merken
                    lines = block["lines"]
                    first_line = lines[0]
                    spans = first_line["spans"]
                    first_span = spans[0]
                    text = first_span["text"]
                    words = word_tokenize(text)
                    first_word = words[0]
                    if(first_word == "Fig") or (first_word == "Tab"):
                        line_description = []
                        for line in lines:
                            span_description = []
                            spans = line["spans"]
                            for span in spans:
                                #if previous_block["type"] == 1:  # description should be directly under the Fig, but sometimes there are words in the Fig, on the bottom
                                    span_description.append(span["text"])
                            line_description.append(span_description)
                        #block_descriptions.append((imgnr, line_description))
                        #print(block_description)
            
                        # 2. die gesplitteten imgs nutzen um ein Bild aus der Seite zu croppen
                        if len(splitted_img_bboxes) > 0:
                            min_x0 = min(splitted_img_bboxes, key = lambda x:x[0])
                            min_y0 = min(splitted_img_bboxes, key = lambda x:x[1])
                            max_x1 = max(splitted_img_bboxes, key = lambda x:x[2])
                            max_y1 = max(splitted_img_bboxes, key = lambda x:x[3])
            
                            page.set_cropbox(fitz.Rect(min_x0[0], min_y0[1], max_x1[2], max_y1[3]))
                            pix = page.get_pixmap(dpi = 500)
                            #print(pix)
                            pixmaps.append(pix)
                            pix.save(os.path.join(workdir, stem + "-img-%i.png" % (imgnr)))
                        
                            blocks_data.append((imgnr, pix.width, pix.height, line_description, page.number +1))
                    
                            imgnr += 1
                            img_found = False
                            splitted_img_bboxes =[]
                        
                    else: splitted_img_bboxes.append(block['bbox'])
            
            
    #print(blocks_data)
    #print(len(blocks_data)) # image count

    #for i in range(1, len(blocks_data)+1):
        #print(i)
        #image = Image.open(os.path.join(workdir,'img-%i.png' % i))
        #image.show()
    
    return blocks_data

def resizeImages(blocks_data, top_k_terms_scores, stem):

    workdir = 'extracted_images'

    scale_max = 0.5    # controls influence of key terms
    resized_data = []

    for block_data in blocks_data:
        found_terms = []
        text = block_data[3]
        for line in text:
            for span in line:
                words = word_tokenize(span)
                for word in words:
                    for term in top_k_terms_scores:
                        if word in term[0]:
                            found_terms.append(term)

        if found_terms:
            w_max = max(found_terms, key = lambda x:x[1])
            scale = scale_max * w_max[1]
                        
        else:
            scale = 0

        aslist = list(block_data)
        aslist[1] *= scale
        aslist[2] *= scale
        resized_data.append(aslist)

    # resize the saved images

    for data in resized_data:
        image = Image.open(os.path.join(workdir, stem + '-img-%i.png' % data[0]))

        if (data[1] != 0) and (data[2] != 0):
            resized_image = image.resize((int(data[1]), int(data[2])))
            resized_image.save(os.path.join(workdir, stem + '-resized_img-%i.png' % data[0]))

        #scale = 0.5
        #size = (int(scale * image.size[0]), int(scale * image.size[1]))
        #scaled_img = image.resize(size)
        #scaled_img.save(os.path.join(workdir, stem + '-resized_img-%i.png' % data[0]))

    return resized_data

def classifyImages(blocks_data, resized_data, stem):
    workdir = 'extracted_images'
    img_classes = []

    for i in range(1, len(resized_data)+1):
    #print(i)
        try:
            image = Image.open(os.path.join(workdir, stem + '-resized_img-%i.png' % i))
            pixels = image.load() 
            width, height = image.size
            img_size = width * height

            all_pixels = []
            white_count = 0
    
            for x in range(width):
                for y in range(height):
                    cpixel = pixels[x, y]
                    #print(cpixel)
            
                    luma = (0.3 * cpixel[0]) + (0.59 * cpixel[1]) + (0.11 * cpixel[2])
                    all_pixels.append(luma)
            
                    if cpixel == (255, 255, 255):
                        white_count += 1

            white_percent = white_count/(width * height)
            #print(white_percent) # wie viel % weiße pixel gibt es
            sum_pixels = np.sum(all_pixels)/(width * height)
            #print(sum_pixels) # je höher desto heller ist das Bild
    
    
            # geeigneten threshold finden und die Bilder in A und B klassifizieren
            if (white_percent >= 0.2) and (sum_pixels >= 150 ): # über 20% weiße Pixel und eine Helligkeit von mind. 150
                img_classes.append((i, "A", img_size, width, height))
            else: img_classes.append((i, "B", img_size, width, height))

        except:
            pass
        
    #print(img_classes)
    return img_classes


def selectImages(img_classes):
    # sort images

    sorted_imgs = sorted(img_classes, key=lambda x: x[2], reverse=True)
    #print(sorted_imgs)


    # take first 4 imgs and look if each class is represented

    classes = []
    selected_imgs = []

    if len(sorted_imgs) < 4:
        img_count = len(sorted_imgs)
    else: img_count = 4

    for i in range(img_count):
        classes.append(sorted_imgs[i][1])
        selected_imgs.append(sorted_imgs[i])

    
    # if a class is not represented, discard last img and take largest img of missing class

    found = False

    if ("A" not in classes):
    
        for img in sorted_imgs:
            if img[1] == "A":
                del classes[3]
                del selected_imgs[3]
                missing_img = img
                found = True
                break;
        
        if found:
            classes.append(missing_img[1])
            selected_imgs.append(missing_img)
    
    
    if ("B" not in classes):
        for img in sorted_imgs:
            if img[1] == "B":
                del classes[3]
                del selected_imgs[3]
                missing_img = img
                found = True
                break;
    
        if found:
            classes.append(missing_img[1])
            selected_imgs.append(missing_img)
    
    #selected_imgs.append((2, 'B', 1234))
    #print(selected_imgs)


    # if an area of an img is smaller than 25% of the largest img -> discard

    filtered_imgs = []

    if selected_imgs:
        largest_img = selected_imgs[0]

        for img in selected_imgs:
            if img[2] >= (0.25 * largest_img[2]):
                filtered_imgs.append(img)
        
        #print(filtered_imgs)
    return filtered_imgs

def getBackgroundColor(filtered_imgs, stem):
    workdir = 'extracted_images'

    h_values = []

    for img in filtered_imgs:
    
        image = Image.open(os.path.join(workdir, stem + '-img-%i.png' % img[0]))
        pixels = image.load() 
        width, height = image.size
    
        for x in range(width):
            for y in range(height):
                rgb_pixel = pixels[x, y]
                hsv_pixel = colorsys.rgb_to_hsv(rgb_pixel[0],rgb_pixel[1],rgb_pixel[2])
                if hsv_pixel[1] != 0.0:
                    h_values.append(hsv_pixel[0])
            

    counts = Counter(h_values) 
    most_freq_h, the_count = counts.most_common(1)[0] 
    #print(most_freq_h)

    def hsv2rgb(h,s,v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

    background_col = hsv2rgb(most_freq_h, 0.1, 1)
    #print(background_col)


    # set background string
    background_col_str = 'rgb('

    i = 0
    for col in background_col:
        if i<2:
            background_col_str += str(col) + ','
        else:
            background_col_str += str(col)
        i += 1
    background_col_str += ')'

    return background_col, background_col_str

def getImagePositions(filtered_imgs):
    img_sizes = []

    for img in filtered_imgs:
        img_sizes.append((img[3]+50, img[4]+50))
    #print(img_sizes)

    img_positions = rpack.pack(img_sizes)
    #print(img_positions)


    # img positions with upper left and bottom right corner
    img_positions_extended = []

    index = 0

    for pos in img_positions:
        x_pos = pos[0]
        y_pos = pos[1]
        width = filtered_imgs[index][3]
        height = filtered_imgs[index][4]
        img_positions_extended.append(((pos), (x_pos + width, y_pos + height)))
        index += 1
    
    return img_positions, img_positions_extended

def getDocCardWidthHeight(img_positions):
    # get the max height and width for the doccard

    max_x = 0
    max_y = 0

    index = 0

    for pos in img_positions:
        endpos = pos[1]

        x = endpos[0]
        y = endpos[1]
    
        # img am weitesten rechts
        if x >= max_x:
            max_x = x
            max_x_img = index
        
        # img am weitesten unten
        if y >= max_y:
            max_y = y
            max_y_img = index
        
        index += 1

    if max_x < 2480:
        max_x = 2480
    
    if max_y < 3508:
        max_y = 3508

    return max_x, max_y

def getFreeSpaceRect(img_positions, max_x, max_y):

    L_i = img_positions
    L_r = [((0,0),(max_x, max_y))]        # free space rectangles, initialized with DC bounding box

    for i in L_i: # für alle image positions
        #print("i: ",i)
        for r in L_r: # für alle freien rectangles
            #print("r: ",r)
        
            # if image is contained in free rect
            if (r[0][0] <= i[0][0]) and (i[1][0] <= r[1][0]) and (r[0][1] <= i[0][1]) and (i[1][1] <= r[1][1]):
            
                #print("yes")
            
                # split r in four new rectangles
                r_T = ((r[0][0], r[0][1]), (r[1][0], i[0][1]))
                #print("r_T: ",r_T)
                r_B = ((r[0][0], i[1][1]), (r[1][0], r[1][1]))
                #print("r_B: ",r_B)
                r_L = ((r[0][0], i[0][1]), (i[0][0], i[1][1]))
                #print("r_L: ",r_L)
                r_R = ((i[1][0], i[0][1]), (r[1][0], i[1][1]))
                #print("r_R: ",r_R)
            
                # add all r_X to L_r
                L_r.append(r_T)
                L_r.append(r_B)
                L_r.append(r_L)
                L_r.append(r_R)
            
                # remove r from L_r
                L_r.remove(r)
            
    return L_r

def sortRectangles(L_r):

    # sort free space rectangles by dec size
    L_r_size = []

    for rect in L_r:
        width = rect[1][0] - rect[0][0]
        height = rect[1][1] - rect[0][1]
        size = (width * height)
        if size > 0:
            L_r_size.append((rect, width, height, size))
        

    L_r_sorted = sorted(L_r_size, key = lambda x: -x[3])

    return L_r_sorted

    
def makeOverview(max_x, max_y, filtered_imgs, L_r, background_col, img_positions):
    
    workdir = 'extracted_images'
    pn.extension()

    # DIN A4 size: 2480 x 3508
    overview = Image.new(mode='RGB', size=(max_x, max_y))

    index = 0
    for img in filtered_imgs:
        #print(os.path.join(workdir,'resized_img-%i.png' % img[0]))
        image = Image.open(os.path.join(workdir,'resized_img-%i.png' % img[0]))
        overview.paste(image, img_positions[index])
    
        index += 1
    
    draw = ImageDraw.Draw(overview)

    for free_space_rect in L_r:
        draw.rectangle([free_space_rect[0], free_space_rect[1]], fill=(background_col[0], background_col[1], background_col[2])) # color has to be the most occuring color

    #pn.pane.Image(overview).show()

    return overview

def getTooltipsPageNumbs(filtered_imgs, resized_data, abstract):

    img_tooltips = []
    img_pages = []

    for filtered_img in filtered_imgs:
        for resized_img in resized_data:
            if filtered_img[0] == resized_img[0]:
                img_tooltips.append(resized_img[3])
                img_pages.append(resized_img[4])
    #print(img_pages)

    tooltips_text = []
    tooltips = []

    for fig_description in img_tooltips:
        description = ''
        for sentence in fig_description:
            description += sentence[0]
        tooltips_text.append(description)
    #print(tooltips_text)

    for text in tooltips_text:
        tooltips.append(Tooltip(content=str(text), position="right", styles={'width': '300px'}))
    #print(tooltips)

    abstract_tooltip = Tooltip(content=str(abstract), position="bottom", styles={'width': '300px'})

    return img_pages, tooltips, abstract_tooltip

def makeImgButtons(filtered_imgs, tooltips, img_positions, stem):

    # problems with css background image
    
    workdir = 'extracted_images'
    pn.extension()

    img_panes = []
    img_buttons = []

    for i,img in enumerate(filtered_imgs):

        url = workdir + '/' + stem + '-resized_img-%i.png' % img[0]
        #print(url)

        pos = img_positions[i]
        pos_x = int(pos[0] * 0.15)
        pos_y = int(pos[1] * 0.15)
    
        changed_width = int(img[3] * 0.15)
        changed_height = int(img[4] * 0.15)
    
        img_pane = pn.pane.Image(f'{url}', width=changed_width, height=changed_height, styles={
            'position': 'absolute',
            'left': f'{pos_x}px', # x abstand zum linken rand
            'top': f'{pos_y}px' # y abstand zum oberen rand
        })

        img_button = pn.widgets.Button(button_type='default', description=tooltips[i], button_style='outline', width=changed_width, height=changed_height, styles={
        'position': 'absolute',
        'left': f'{pos_x}px', # x abstand zum linken rand
        'top': f'{pos_y}px' # y abstand zum oberen rand
        })

        img_panes.append(img_pane)
        img_buttons.append(img_button)

    #print(img_buttons)

    return img_buttons, img_panes


def find_top_k_terms_docCard(n, max_x, max_y, filtered_imgs, terms_and_scores):
   
    # max number of key terms =  n

    a_dc = max_x * max_y
    a_img = 0


    for img in filtered_imgs:
        width = img[3]
        height = img[4]
        size = width * height
        a_img += size

    k = math.floor(n*((a_dc-a_img)/a_dc))
    #print("k: ", k)

    top_k_terms, top_k_terms_scores = find_top_k_terms(k, terms_and_scores)

    #print(top_k_terms)
    #print(top_k_terms_scores)

    return top_k_terms, top_k_terms_scores

def posTerms(top_k_terms_scores_docCard, sorted_rectangles):

    #print("terms to place: ", top_k_terms_scores_docCard)
    #print("sorted rectangles: ", sorted_rectangles)
    
    term_panes = []
    str_buttons = []
    term_sizes_positions = []

    s_min = 80 # min font size
    s_max = 120 # max font size

    w_min = 0.3

    beta = 1-(s_min/s_max)



    def place_terms(beta, terms_to_place, L_r_sorted):
    
        #print("terms to place: ", terms_to_place)
        #print("sorted rectangles: ", L_r_sorted)
        #print("beta: ", beta)
    
        if (beta > 0) and (terms_to_place) and (L_r_sorted): # wenn beta noch größer 0 ist und noch terme und rectangles übrig sind
        # iterating terms and free space rectangles to position
    
            for term in terms_to_place:
                #print("aktueller term: ", term)
            
                # zuerst die Größe des Terms bestimmen, damit man schauen kann, ob er in ein Rechteck passt
            
                # calculate font size
                w_i = term[1]
                scale = (s_min/s_max) + beta * ((w_i-w_min)/(1.0-w_min))
                #print("scale: ", scale)
                font_size = s_max * scale
                #print("font_size: ",font_size)
        
                # berechne Länge und Höhe des Text Panes, in welches der Term platziert wird
                text = term[0][0]
                text_pane_width = int(len(text)*font_size)
                text_pane_height = int(font_size)
                #print("term_pane width: ", text_pane_width)
                #print("term_pane height: ", text_pane_height)
        
                # jedes rectangle durchgehen und schauen, ob der Term rein passt
                for rect in L_r_sorted:
                    #print("aktuelles rectangle: ", rect)
                    rect_width = rect[1]
                    rect_height = rect[2]
            
                    # wenn der Term in das aktuelle rectangle rein passt
                    if (text_pane_width < rect_width) and (text_pane_height < rect_height):
                        #print("term fits in rectangle!")
                
                
                        # Berechne die Position wo das Text Pane platziert wird
                        # calculate center position of the rectangle
                        rect_center_x = (rect[0][0][0] + rect[0][1][0])/2
                        rect_center_y = (rect[0][0][1] + rect[0][1][1])/2
                    
                        # calculate start and end pos of the text pane
                        x_start = rect_center_x-(text_pane_width/2)
                        y_start = rect_center_y-(text_pane_height/2)
                        x_end = rect_center_x+(text_pane_width/2)
                        y_end = rect_center_y+(text_pane_height/2)
                        #center_pos = (rect_center_x, rect_center_y)
                
                        # save font size and start position for term
                        term_sizes_positions.append((term[0], font_size, (x_start, y_start)))
                    
                
                        # make str_pane and adjust to fit in the scaled rectangle
                        str_pane = pn.widgets.StaticText(
                        value = text,
                        width = int(text_pane_width*0.15),
                        height = int(text_pane_height*0.15),
                        styles={'font-size': f'{int(font_size*0.15)}pt',
                            'position': 'absolute', 
                            'top': f'{int(y_start*0.15)}px', 
                            'left': f'{int(x_start*0.15)}px',
                            #'background-color': 'lightblue',
                            'text-align': 'center'}
                        )
                        term_panes.append(str_pane) # store to use later


                        # create buttons for clicking on the key words
                        str_button = pn.widgets.Button(button_type='default', value=False, button_style='outline', width=int(text_pane_width*0.15), height=int(text_pane_height*0.3), styles={
                            'position': 'absolute',
                            'left': f'{int(x_start*0.15)}px',
                            'top': f'{int(y_start*0.15)}px'
                        })
                
                        str_buttons.append(str_button)


                    
                        # Nachdem der Term platziert wurde, muss das alte Rechteck gelöscht werden, damit es nicht doppelt benutzt wird
                        # und neue freie Rechtecke müssen berechnet werden
                    
                        # split rectangle like before and remove the old rectangle form L_r
                        pane_rect = ((x_start, y_start), (x_end, y_end)) # the text pane rectangle
                
                        new_rectangles = []
                        r_T = ((rect[0][0][0], rect[0][0][1]), (rect[0][1][0], pane_rect[0][1]))
                        new_rectangles.append(r_T)
                        r_B = ((rect[0][0][0], pane_rect[1][1]), (rect[0][1][0], rect[0][1][1]))
                        new_rectangles.append(r_B)
                        r_L = ((rect[0][0][0], pane_rect[0][1]), (pane_rect[0][0], pane_rect[1][1]))
                        new_rectangles.append(r_L)
                        r_R = ((pane_rect[1][0], pane_rect[0][1]), (rect[0][1][0], pane_rect[1][1]))
                        new_rectangles.append(r_R)
                        #print("die neuen Rs: ", new_rectangles)
            
                        # add all r_X to L_r_sorted and calculate the values 
                        for new_rect in new_rectangles:
                            width = new_rect[1][0] - new_rect[0][0]
                            height = new_rect[1][1] - new_rect[0][1]
                            size = (width * height)
                            if size > 0:
                                L_r_sorted.append((new_rect, width, height, size))
                
                        # remove rect
                        #print("rect to remove: ", rect)
                        L_r_sorted.remove(rect) # das aktuelle rectangle wird nicht mehr für den Rest gebraucht und daher entfernt
                        L_r_sorted = sorted(L_r_sorted, key = lambda x: -x[3]) # nochmal L_r_sorted sortieren, da neue Rechtecke dazu gekommen sind
                        #print("L_r_sorted without fitting rectangle: ", L_r_sorted)
                
                        # remove term
                        #print("term to remove: ", term)
                        terms_to_place.remove(term) # term wurde platziert und wird nicht mehr für den Rest gebraucht und daher entfernt
                        #print("term list after removing: ", terms_to_place)
                    
                        # wenn ein pasendes rectangle gefunden wurde, wird die for-Schleife verlassen und der nächste term ist dran
                        break;
                
                    # wenn der Term nicht in das rectangle passt
                    #
                    # else: 
                        #print("term does not fit in this rectangle!")
        
            # alle terme sind mit dem aktuellen beta durch -> beta wurd nun kleiner gemacht und die Fkt so lange aufgerufen 
            # bis wir im else landen und die positionen und panes zurückgeben
            beta -= 0.1
            place_terms(beta, terms_to_place, L_r_sorted)
        
        
        
        #else:    # wenn beta kleiner 0 oder es keine terme oder rectangles mehr gibt -> fertig
        
            #if (not term_sizes_positions) or (not term_panes):
                #print("no term fitted in any rectangle")
    
            #return term_sizes_positions, term_panes
    
    
    
    place_terms(beta, top_k_terms_scores_docCard, sorted_rectangles)

    return term_sizes_positions, term_panes, str_buttons

def countTermAmount(pages_text, term_sizes_positions):
    terms_amount = []

    for term_info in term_sizes_positions:
        page_number = 1
        for page_text in pages_text:
            count = page_text.count(term_info[0][0])
            terms_amount.append((term_info[0][0], count, page_number))
            page_number += 1
    #print("terms amount: ", terms_amount)
            
    min_of_terms = 0
    max_of_terms = max(terms_amount, key=lambda term: term[1])

    # normalize term amount
    terms_amount_norm = []

    min_amount = min_of_terms
    max_amount = max_of_terms[1]

    for term in terms_amount:
        amount = term[1]
        amount_norm = (amount - min_amount) / (max_amount - min_amount)
        terms_amount_norm.append((term[0], amount_norm, term[2]))
    #print(terms_amount_norm)
        
    return terms_amount_norm, terms_amount, min_amount, max_amount

def getTermColors(terms_amount_norm, terms_amount, min_amount, max_amount):

    norm_words = mpl.colors.Normalize(vmin=min_amount, vmax=max_amount)
    cmap_words = cm.RdPu
    m = cm.ScalarMappable(norm=norm_words, cmap=cmap_words)

    terms_colors = []

    for term in terms_amount:
        color = m.to_rgba(term[1])
        rgb = []
        for value in color:
            value *= 255
            rgb.append(value)
        terms_colors.append((term[0], rgb, term[2]))

    return terms_colors


def createDocCard(doc, authors, title, background_col, background_col_str, abstract_tooltip, max_x, max_y, L_r, img_buttons, img_panes, term_panes, str_buttons, terms_colors, img_pages, stem):
    pn.extension()

    page_index = 0
    button_list = []
    button_count = len(doc)



    # buttons on the left
    overview_button = pn.widgets.Button(name='Overview', description=abstract_tooltip, value=True, button_type='default', button_style='outline', width=77, height=32, styles={'background': background_col_str, 'border-radius': '5px', 'font-family': 'Tahoma, sans-serif' '!important'})

    for page_index in range(1, button_count+1):
        page_button = pn.widgets.Button(name=str(page_index), value=False, button_type='default', button_style='outline', width=77, height=32, styles={'background': 'white', 'border-radius': '5px', 'font-family': 'Tahoma, sans-serif' '!important'})
        button_list.append(page_button)

    pageNav = pn.Column(
            overview_button, 
            *button_list)
    
    
    
    # free space rectangles
    im = Image.new(mode='RGB', size=(int(max_x*0.15), int(max_y*0.15)), color=(background_col[0], background_col[1], background_col[2]))
    draw = ImageDraw.Draw(im)

    for free_space_rect in L_r:
    
        start_pos = free_space_rect[0]
        end_pos = free_space_rect[1]
    
        x1 = start_pos[0]*0.15
        y1 = start_pos[1]*0.15
        x2 = end_pos[0]*0.15
        y2 = end_pos[1]*0.15
    
        draw.rectangle([(x1,y1), (x2,y2)], fill=(background_col[0], background_col[1], background_col[2]))


    
    # handle buttons
    mainbox_buttons = button_list + img_buttons + str_buttons
    buttonbox_buttons = button_list + img_buttons + str_buttons


    def main_box(overviewButtonValue, *buttonValues):

        if overviewButtonValue:

            for button in str_buttons:
                button.button_type='default'

            for button in button_list:
                    button.styles={'background': 'white'}
            overview_button.styles={'background': background_col_str}

            return overview
        
        for i, val in enumerate(buttonValues):
            if val:
                if i < len(button_list): # page button clicked
                    return pn.pane.Image(f'extracted_pages/{stem}-page-{i+1}.png', width=int(max_x*0.15), height=int(max_y*0.15))

                if i >= len(button_list) and i < (len(button_list) + len(img_buttons)): # img buttons clicked
                    pageNumber = img_pages[i-len(button_list)]
                    return pn.pane.Image(f'extracted_pages/{stem}-page-{pageNumber}.png', width=int(max_x*0.15), height=int(max_y*0.15))

                if i >= (len(button_list) + len(img_buttons)): # key word buttons clicked
                    for button in str_buttons:
                        button.button_type='default'
                    clicked_word_index = i-(len(button_list) + len(img_buttons))
                    clicked_button = mainbox_buttons[i]
                    clicked_button.button_type='warning'
                    ov = pn.Column(pn.pane.Image(im),
                            *(img_panes),
                            *(img_buttons), 
                            *(term_panes), 
                            *(str_buttons),
                            width = int(max_x*0.15), height = int(max_y*0.15) )
                    return ov
        return overview 


    def button_box(overviewButtonValue, *buttonValues):

        if overviewButtonValue:
            return pageNav
        
        for i, val in enumerate(buttonValues):
            if val:
                for button in button_list:
                    button.styles={'background': 'white'}

                if i < len(button_list): # page buttons clicked
                    overview_button.styles={'background': 'white'}
                    clicked_button = button_list[i]
                    clicked_button.styles={'background': background_col_str}
                    nav = pn.Column(
                            overview_button, 
                            *button_list)
                    return nav

                if i >= len(button_list) and i < (len(button_list) + len(img_buttons)): # img buttons clicked
                    overview_button.styles={'background': 'white'}
                    pageNumber = img_pages[i-len(button_list)]
                    clicked_button = button_list[pageNumber-1]
                    clicked_button.styles={'background': background_col_str}
                    navi = pn.Column(
                            overview_button, 
                            *button_list)
                    return navi
                
                if i >= (len(button_list) + len(img_buttons)): # key word buttons clicked
                    clicked_word_number = i-(len(button_list) + len(img_buttons)) + 1
                    sliced_terms_colors = terms_colors[((clicked_word_number-1)*len(button_list)):(clicked_word_number*len(button_list))]

                    for i, button in enumerate(button_list):
                        term_page_i = sliced_terms_colors[i]
                        col_term_page_i = term_page_i[1]
                        rgb = []
                        rgb.append(col_term_page_i[0]) # r
                        rgb.append(col_term_page_i[1]) # g
                        rgb.append(col_term_page_i[2]) # b

                        rgb_str = 'rgb('
                        i = 0
                        for col in rgb:
                            if i < 2:
                                rgb_str += str(col) + ','
                            else:
                                rgb_str += str(col)
                            i += 1
                        rgb_str += ')'
                        button.styles={'background': rgb_str}

                    navigation = pn.Column(
                            overview_button, 
                            *button_list)
                    return navigation

        return pageNav


    
    # create Document Card
    overview = pn.Column(pn.pane.Image(im),
                        *(img_panes),
                        *(img_buttons),
                        *(term_panes), 
                        *(str_buttons),
                        width = int(max_x*0.15), height = int(max_y*0.15) )

    docCard = pn.Row(pn.Column(pn.widgets.StaticText(value=title, width = int(max_x*0.15), styles={'background': 'white', 'font-size': '12pt', 'border': '2px solid ' + background_col_str, 'border-radius': '15px', 'padding': '20px', 'font-family': 'Tahoma, sans-serif', 'text-align': 'center'}),
                        pn.bind(main_box, overview_button, *mainbox_buttons), 
                        pn.widgets.StaticText(value=authors, width = int(max_x*0.15), styles={'background': 'white', 'font-size': '12pt', 'border': '2px solid ' + background_col_str, 'border-radius': '15px', 'padding': '20px', 'font-family': 'Tahoma, sans-serif', 'text-align': 'center'})), 
                    pn.Column(pn.Row(pn.Spacer(styles=dict(background='yellow'))), pn.bind(button_box, overview_button, *buttonbox_buttons), pn.Row(pn.Spacer(styles=dict(background='blue')))))


    return docCard

main()