# Author(s): Kenneth Foo, Brayden Tang, Brendon Campbell
# Date: January 22, 2020 

"This script splits the raw data into train and test sets, This script
assumes that it is being run from the root directory of the repository.

Usage: wrangling.R <file_raw> <path_out>

Options:
<file_raw>   A file path that gives the location of the raw data.
<path_out>   A file path specifying where to store train.csv and test.csv.
" -> doc

library(tidyverse)
library(caret)
library(testthat)
library(docopt)

#' This function takes the raw data from a specified path and
#'  preprocesses it by dropping the highly correlated features G1 and G2, 
#'  and then splitting the data into train and test sets.
#'
#' @param file_raw
#'  A character vector of length one that provides the exact file path to a .csv
#'  file containing the raw data. The file path should be a relative from the
#'  root of the repository.
#' @param path_processed 
#'  A character vector of length one that gives the location of where the 
#'  processed train and test set sets will be stored. The files outputted 
#'  by this script end with train.csv and test.csv respectively. This path
#'  should be defined relate to the root of the repository.
#'
#' @return NA
#' @export
#'
#' @examples
#' main(
#'  file_raw = "data/raw/student-por.csv", 
#'  path_processed = "data/processed")
#' 
main <- function (file_raw, path_processed) {

if (str_detect(file_raw, ".csv") == FALSE) {
  stop("The raw file path must be a .csv file.")
} 

if (str_detect(path_processed, ".csv") == TRUE) {
  stop("The path to store the processed train and test sets should just
  be a file path and not a file.")
}
  
df <- read_delim(file_raw, delim = ";")

set.seed(200350623)
split <- caret::createDataPartition(y = df$G3, times = 1, p = 0.8)

train_df <- df[split[[1]], ] 

test_df <- df[-split[[1]], ] 

write_csv(train_df, paste(path_processed, "/train.csv", sep = ""))
write_csv(test_df,  paste(path_processed, "/test.csv", sep = ""))

paste(
  "Train and test sets stored in ",
  path_processed, 
  "/train.csv and ",
  path_processed, "/test.csv", sep = "")

}

#' This function tests main for invalid file names or 
#' invalid file paths.
#'
#' @return
#'  A character string that outputs "All tests have passed." 
#'  if all tests have passed.
#' @export
#'
#' @examples tester()
tester <- function() {
  
  test_that("The function accepts a file that isn't .csv", {
    expect_error(main("data/raw/student-por", "data/processed"))
    expect_error(main("cool/student-por", "data/processed"))
  })
  
  test_that("The function accepts a file name as the output 
  directory when it should only accept a file path.", {
    expect_error(main("data/raw/student-por.csv", "data/processed/train.csv"))
    expect_error(main("data/raw/student-por.csv", "data/processed/coolname.csv"))
  })
  
  paste("All tests have passed.")
  
}

tester()
opt <- docopt(doc)

main(opt$file_raw, opt$path_out)
