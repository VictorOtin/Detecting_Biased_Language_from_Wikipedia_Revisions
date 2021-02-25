import gzip
from xml.etree import ElementTree as ET
import re
import difflib
import time
import sys

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

def gz_to_ids(gz_filepath, search_regex=None):
    counter = 0
    if not search_regex:
        # Use search regex from Pryzant2019
        search_regex = '([- wnv\/\\\:\{\(\[\"\+\'\.\|\_\)\#\=\;\~](rm)?(attribute)?(yes)?(de)?n?pov)|([- n\/\\\:\{\(\[\"\+\'\.\|\_\)\#\;\~]neutral)'
    outpath = gz_filepath[:-6]+'ids' # Replace .xml.gz with .ids
    with open(outpath,'w') as outfile:
        # Write search regex to top of file
        outfile.write('###\nSearch Regex: \"{}\"\nData is in format <page_id> \\t <revision_id> \\t <parent_id>\n###\n'.format(search_regex))
        with gzip.open(gz_filepath,'rt') as infile:
            for event, page in ET.iterparse(infile):
                if page.tag == _page:
                    first_rev_in_page_flag = True
                    for revision in page.findall(_revision):
                        comment = revision.find(_comment)
                        if comment is not None:
                            # Look for search regex (case-insensitive)
                            match = re.search(search_regex, str(comment.text), re.I)
                            if match:
                                if first_rev_in_page_flag:
                                    parentid_elem = revision.find(_parentid)
                                    if parentid_elem is None:
                                        # parent_id doesnt exist if its the first revision of the page
                                        continue
                                    page_id = page.find(_id).text
                                    outfile.write('\n'+page_id) # Create new line for this page
                                    first_rev_in_page_flag = False

                                rev_id = revision.find(_id).text
                                par_id = parentid_elem.text
                                
                                outfile.write('\t'+rev_id+'\t'+par_id)
                                counter += 1
                                if counter%100 == 0:
                                    print("Found {} revisions".format(counter))
                    
                    # Clear up memory
                    page.clear()
    print("Found {} revisions in total".format(counter))
    return outpath

if __name__ == "__main__":
    url = sys.argv[1]
    gz_file = download_file(url,'../../data/raw/WPRH/meta')
    start = time.time()
    ids_file = gz_to_ids(gz_file)
    end = time.time()
    print('Elapsed time in seconds:{}'.format(end-start))
    print(ids_file)

