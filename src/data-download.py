# Author(s): Kenneth Foo, Brayden Tang, Brendon Campbell
# Date: January 17, 2020 

"""This script downloads and extracts a .zip file from a specified 
web URL to a given file path provided by the user. This script assumes that it will be run
in the root of the project directory.

Usage: data-download.py <url_link> --file_path=<file_path>

Options:
<url_link>                  A url link to the .zip file.
--file_path=<file_path>     A relative file path to which the .zip file will be downloaded.
"""

import requests, zipfile, io, pytest
from docopt import docopt


opt = docopt(__doc__)


def main(url_link, file_path):
    """ 
    This function downloads and extracts a .zip file from a valid url
    link and stores them in a given file path.
    
    Parameters
    ----------
    url_link: str
        A str that gives a web URL to a .zip file to be downloaded. The link must be
        valid, or else an error will be thrown.
    
    file_path: str
        A str that provides an absolute file path in which the extracted .zip file will
        be stored. Cannot be null, otherwise an error will be thrown.
    
    Returns
    ---------
    None
    
    Examples
    ---------
    main(
        url_link="https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
        file_path="data/raw"
    )

    """
    
    if "https://" not in url_link or ".zip" not in url_link or not url_link:
        raise Exception("Invalid url_link argument.")
    
    if not file_path:
        raise Exception("Please provide a valid file path.")
    
    r = requests.get(url_link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # Assume run cmd line from project repo root. 
    z.extractall("./" + file_path)
    print("File downloaded to:", file_path)
    return 

def check_valid_url():
    """
    This function checks if an exception is raised if invalid url_links are provided to 
    main.
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    None, if the test has passed, and a Failed message if the test has not passed.
    
    Examples
    ----------
    check_valid_url()
    
    """
    with pytest.raises(Exception):
        main("abc.com", "data")
        main("https://archive.ics.uci.edu/ml/machine-learning-databases/00320", "data")

def check_file_path():
    """
    This function checks if an exception is raised if an empty file_path is provided to 
    main.
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    None, if the test has passed, and a Failed message if the test has not passed.
    
    Examples
    ----------
    check_file_path()
    
    """
    with pytest.raises(Exception):
        main("https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip")
        
assert check_valid_url() == None, "Invalid url_link check of main function has failed."
assert check_file_path() == None, "Invalid file path check of main function has failed."

# Pull in the data
main(url_link=opt['<url_link>'], file_path=opt['--file_path'])
