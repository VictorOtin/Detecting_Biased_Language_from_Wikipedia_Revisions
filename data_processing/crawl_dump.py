import bz2
import gzip
from xml.etree import ElementTree as ET
import re
import difflib
import time
from datetime import datetime
import importlib
import sys
import os

import mwparserfromhell
import py7zr

from download_data import download_file

# XML Element tag strings
_tag_prefix = """{http://www.mediawiki.org/xml/export-0.10/}"""
_page = _tag_prefix + "page"
_title = _tag_prefix + "title"
_ns = _tag_prefix + "ns"
_id = _tag_prefix + "id"
_parentid = _tag_prefix + "parentid"
_revision = _tag_prefix + "revision"
_comment = _tag_prefix + "comment"
_text = _tag_prefix + "text"



def get_id_struct(ids_filepath, min_page_id, max_page_id):
    """
    From the crawled ids (Step one of our two-step process),
    create a list of page-ids and the corresponding revision (and parent revision) ids
    """
    id_struct = []
    with open(ids_filepath, 'r') as ids_file:
        # Ignore first 5 lines (meta info)
        next(ids_file)
        next(ids_file)
        next(ids_file)
        next(ids_file)
        next(ids_file)

        for line in ids_file:
            page_id, rev_id, par_id = line.rstrip().split('\t')[:3]
            if int(page_id) < int(min_page_id): 
                continue
            if int(page_id) > int(max_page_id):
                return id_struct
            else:
                id_struct.append([page_id,[rev_id,par_id]])
        return id_struct


def crawl_multiple_files(start_index, end_index, ids_filepath, print_info):
    filenames = []
    # Names of all dumpfiles
    with open('../../data/other/dump_filenames.txt') as dumpnames_file:
        for line in dumpnames_file:
            filenames.append(line.split()[-3])

    dump_dir = "https://dumps.wikimedia.org/enwiki/20200501/"
    
    # Process
    for index,filename_7z in enumerate(filenames[start_index:end_index]):
        url = dump_dir+filename_7z
        dump_filepath = download_file(url,'../../data/raw/WPRH/full')
        filename = filename_7z[:-3]
        outpath = '../../data/processed/crawl/'+filename+'.diffs'

        start = time.time()
        diffs_from_file(dump_filepath, filename, ids_filepath, outpath, print_info)
        end = time.time()
        print('Dump index: {}, \tElapsed time in seconds:{}'.format(start_index+index,end-start))
        print(outpath)


def diffs_from_file(dump_filepath, filename, ids_filepath, out_filepath, print_info):
    """
    Creates the list of page_ids for a given dump file,
    crawls that dump file,
    and removes dump file from memory
    """
    low_page_id, high_page_id = re.findall(r'p(\d+)',filename)
    
    id_struct = get_id_struct(ids_filepath, low_page_id, high_page_id)
    
    crawl_single_file(dump_filepath, out_filepath, id_struct, print_info)
    
    # Remove .7 file
    os.remove(dump_filepath)
    # Remove unzipped file
    os.remove('../../data/raw/WPRH/full/unzipped/'+filename)

    return out_filepath


def crawl_single_file(dump_filepath, out_filepath, id_struct, print_info):
    """
    Outer method for crawl_xml_file()
    Opens a zipped dump file (either .bz2 or .7z)
    and delegates handling of the file contents to crawl_xml_file()
    """
    # .bz2 file
    if dump_filepath.endswith('.bz2'):
        with bz2.open(dump_filepath,'rt') as dumpfile:
            return crawl_xml_file(dumpfile, out_filepath, id_struct, print_info)
    # .7z file
    elif dump_filepath.endswith('.7z'):
        filename = dump_filepath.split('/')[-1][:-3]
        with py7zr.SevenZipFile(dump_filepath, 'r') as zip_file:
            unzipped_dir = '../../data/raw/WPRH/full/unzipped/'
            zip_file.extract(unzipped_dir)
        with open(unzipped_dir+filename, 'r') as dumpfile:
            return crawl_xml_file(dumpfile, out_filepath, id_struct, print_info)



def crawl_xml_file(dumpfile, out_filepath, id_struct, print_info):
    """
    Inputs:
        dumpfile:
            Input dump file
        out_filepath:
            Contains the diffs
        id_struct: 
            List of page_ids, for each a tuple of revision_id and parent_id
            [page_id, [(rev_id, parent_id)...]]
        print_info:
            Boolean, prints extra information like time spent on diff and discarded revisions
    Outputs:
        Writes to outfile in format: <revision_id> tab <parent diff> tab <revision diff>
    """
    page_index = 0
    page_found_flag = False # Signals if a page is in the page_ids
    page_id_flag = False # Signals if its a page id or another id

    with open(out_filepath,'w') as outfile, open('../../data/processed/crawl/log.txt','a') as logfile:
        for line in dumpfile:
            line = line.strip()
            if line == '<page>':
                # Found all pages
                if page_index >= len(id_struct): 
                    break
                else: 
                    # Current page id to look for
                    cur_page_id = id_struct[page_index][0]

                page_id_flag = True
                page_found_flag = False
                page_string = "<page>" # Reset page text
            if line.startswith("<id>") and page_id_flag:
                page_id_flag = False
                if line == '<id>{}</id>'.format(cur_page_id):
                    page_found_flag = True
                    
                    
            if page_found_flag: 

                page_string += "\n" + line # Build page string

                if line == '</page>':
                    try:
                        page = ET.fromstring(page_string)
                    except Exception as exc:
                        logfile.write("Exception parsing page_string as xml tree: {}\n".format(exc))
                        print("Exception parsing page_string as xml tree: {}\n".format(exc))

                    # Revision element of the current revision
                    rev = page.find("./revision[id='{}']".format(id_struct[page_index][1][0]))
                    # Revision element of the parent revision
                    par = page.find("./revision[id='{}']".format(id_struct[page_index][1][1]))

                    if rev is None:
                        if print_info:
                            print('No element with rev_id found in page. Page_id: {} \t Rev_id: {}'.format(cur_page_id, id_struct[page_index][1][0]))
                        logfile.write('No element with rev_id found in page. Page_id: {} \t Rev_id: {}'.format(cur_page_id, id_struct[page_index][1][0]))
                        continue

                    if par is None:
                        if print_info:
                            print('No element with par_id found in page. Page_id: {} \t Par_id: {}'.format(cur_page_id, id_struct[page_index][1][1]))
                        logfile.write('No element with par_id found in page. Page_id: {} \t Par_id: {}'.format(cur_page_id, id_struct[page_index][1][1]))
                        continue


                    rev_text = rev.find("text").text
                    par_text = par.find("text").text

                    time_diff_s = time.time()
                    diff = diff_revisions(par_text,rev_text,print_info)
                    time_diff_e = time.time()

                    if diff is not None:
                        outfile.write(str(id_struct[page_index][1][0])+'\t'+diff[0]+'\t'+diff[1]+'\n')
                        if print_info:
                            print("Diff took {} seconds for:".format(time_diff_e - time_diff_s))
                            print("\tPage_id: {} \t Rev_id: {} \t Par_id: {}\n".format(cur_page_id, id_struct[page_index][1][0], id_struct[page_index][1][1]))

                    else:
                        logfile.write("Diff None. Page_id: {} \t Rev_id: {} \t Par_id: {}\n".format(cur_page_id, id_struct[page_index][1][0], id_struct[page_index][1][1])) 
                        if print_info:
                            print("diff_revision() returned None:")
                            print("\tPage_id: {} \t Rev_id: {} \t Par_id: {}\n".format(cur_page_id, id_struct[page_index][1][0], id_struct[page_index][1][1]))
                    
                    page_index += 1

    return out_filepath


def diff_revisions(a,b,print_info):
    min_diff_length = 2 # Discard anything less than this
    max_diff_length = 600
    max_context = 300

    # Parse from wikitext to plaintext
    a = mwparserfromhell.parse(a).strip_code()
    b = mwparserfromhell.parse(b).strip_code()

    # a is old revision, b is the new one
    blocks = difflib.SequenceMatcher(None, a, b).get_matching_blocks() # Last one is dummy block
    
    # DEBUG
    #if print_info:
    #    for block in blocks:
    #        print(str(block[0])+'\t'+str(block[1])+'\t'+str(block[2]))

    if len(blocks) < 3:
        if print_info:
            print("Less than 3 matching blocks. Returning no diff.")
        return None

    fb_idx = 0
    # Set first block (usually index 0)
    for i in range(len(blocks)-1):
        if blocks[i+1][0] - (blocks[i][0]+blocks[i][2]) >= min_diff_length or \
            blocks[i+1][1] - (blocks[i][1]+blocks[i][2]) >= min_diff_length:
            fb_idx = i # First block index
            break 
    
    lb_idx = -2
    # Set last block (usually index -2)
    for i in range(len(blocks)-1,0,-1):
        if blocks[i][0] - (blocks[i-1][0]+blocks[i-1][2]) >= min_diff_length or \
            blocks[i][1] - (blocks[i-1][1]+blocks[i-1][2]) >= min_diff_length:
            lb_idx = i # Last block index
            break    
    
    # Length of sequence between last and first matching block (ie length of diff)
    length_a = blocks[lb_idx][0] - blocks[fb_idx][2]
    length_b = blocks[lb_idx][1] - blocks[fb_idx][2]

    # Discard if two diffs too far apart
    if length_a > max_diff_length or length_b > max_diff_length:
        if print_info:
            print("Diff is too long. Returning no diff.")
        return None

    # Find index of first full stop before and after the diff
    # For revision a
    i1 = blocks[fb_idx][0]+blocks[fb_idx][2]
    context = a[i1:i1-max_context:-1]
    back_index = re.search(r' \.|$|[A-Z]\.',context,re.MULTILINE).start()
    if back_index == -1:
        # No full stop found
        i1_stop = i1-max_context
    else:
        i1_stop = i1-back_index

    i2 = blocks[lb_idx][0]
    context = a[i2:i2+max_context]
    forward_index = re.search(r'\. |$|\.[A-Z]',context,re.MULTILINE).start()
    if forward_index == -1:
        i2_stop = i2+max_context
    else:
        i2_stop = i2+forward_index+1
        
    # For revision b
    j1 = blocks[fb_idx][1]+blocks[fb_idx][2]
    context = b[j1:j1-max_context:-1]
    back_index = re.search(r' \.|$|[A-Z]\.',context,re.MULTILINE).start()
    if back_index == -1:
        j1_stop = j1-max_context
    else:
        j1_stop = j1-back_index

    j2 = blocks[lb_idx][1]
    context = b[j2:j2+max_context]
    forward_index = re.search(r'\. |$|\.[A-Z]',context,re.MULTILINE).start()
    if forward_index == -1:
        j2_stop = j2+max_context
    else:
        j2_stop = j2+forward_index+1
    
    return a[i1_stop+1:i2_stop], b[j1_stop+1:j2_stop]    


if __name__ == "__main__":
    ids_filepath = sys.argv[1]

    start_index = int(sys.argv[2])
    end_index = int(sys.argv[3])

    print_info = False
    if len(sys.argv) > 4 and sys.argv[4] == "True":
        print_info = True
    
    start = time.time()
    crawl_multiple_files(start_index, end_index, ids_filepath, print_info)
    end = time.time()
    print('{} dumpfiles processed({}-{}). Total time (incl. downloading): {}'.format(end_index-start_index, start_index,end_index, end-start))

