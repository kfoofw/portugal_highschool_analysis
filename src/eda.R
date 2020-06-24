# Authors: Brayden Tang, Kenneth Foo, Brendon Campbell
# Date: January 24, 2020

"This script creates box plots of all of the predictors, a correlation matrix,
and a plot of the estimated distribution of G3. This script assumes that it will
be run in the root directory of the repository.

Usage: eda.R <path_data> <directory_out>
  
Options:
<path_data>       A file path that gives the location of the data set that will be used to fit each graph.
<directory_out>   A file path specifying where to store all of the rendered graphs.
" -> doc

library(tidyverse)
library(ggcorrplot)
library(ggridges)
library(testthat)
library(docopt)

#' This function creates a bunch of plots (boxplots, a histogram, a correlation matrix, and a ridgeplot)
#' for further use in the report. 
#'
#' @param path_data 
#' A character string that gives the location of the data set that will be used to fit each
#' graph. This path should be relative to the root directory of this repository.
#' @param directory_out 
#' A character string that provides a file path specifying where to store all of the rendered graphs.
#' This path should be relative to the root directory of this repository.
#' @return A character string displaying a message if the plots were successfully rendered.
#' @export
#'
#' @examples
#' main(path_data = "data/processed/train.csv", directory_out = "img")
#' 
main <- function(path_data, directory_out) {
  
train_data <- read_csv(path_data)

if (str_detect(path_data, ".csv") == FALSE) {
  stop("The path to the data should be a .csv file.")
}

if (str_detect(directory_out, ".csv") == TRUE) {
  stop("The output directory should not be a file.")
}

# Correlation Matrix

correlation_mat <- cor(train_data %>% select_if(is.numeric))

ggsave(
  ggcorrplot::ggcorrplot(
    correlation_mat, 
    show.diag = FALSE,
    type = "lower",
    title = "Correlation Plot of All Features (Pearson's R)"), 
  filename = paste(directory_out, "/correlation_matrix.png", sep = ""))

plotting_data <- train_data %>%
  select(failures, Medu, Fedu, Dalc, Walc, school, G3) %>%
  gather(key = predictor, value = value, -G3) %>%
  filter(!predictor %in% c("G1", "G2", "absences")) 

  ggsave(ggplot(data = plotting_data, aes(x = value, y = G3)) +
           facet_wrap(. ~ predictor, scale = "free", ncol = 3) +
           geom_boxplot() +
           coord_flip() +
           theme_minimal() +
           theme(axis.title.y = element_blank()) +
           labs(y = "G3 (Final Portuguese Score Achieved)"),
       filename = paste(directory_out, "/box-plots.png", sep = ""))

# Absences

ggsave(train_data %>%
  filter(G3 != 0) %>%
  mutate(G3 = as.factor(G3)) %>% 
  ggplot(., aes(x = absences, y = G3, group = G3)) +
  geom_density_ridges(fill = "mediumseagreen") +
  theme_ridges() + 
  theme(legend.position = "none") + 
  labs(y = "G3 (Final Portuguese Score Achieved)",
       x = "Absences",
       group = "G3 (Final Portuguese Score Achieved)"), 
filename = paste(directory_out, "/absences.png", sep = ""))

# G3 Distribution

ggsave(ggplot(data = train_data, aes(x = G3, y = ..density..)) +
  geom_histogram(binwidth = 1) +
  theme_minimal() +
  annotate("label", x = 3, y = 0.13, label = paste(
    "Mean:",
    round(mean(train_data$G3), 2), 
    "\n",
    "Median:",
    round(median(train_data$G3), 2),
    "\n",
    "Standard Deviation:",
    round(sd(train_data$G3), 2))) +
    geom_vline(aes(xintercept = quantile(train_data$G3, 0.1), color = "10th")) +
    geom_vline(aes(xintercept = quantile(train_data$G3, 0.9), color = "90th")) + 
    labs(color = "Percentile",
         x = "G3 (Final Portuguese Score Achieved)",
         y = "Frequency (normalized to 1)"),
  filename = paste(directory_out, "/g3_hist.png", sep= ""))

paste("Plots successfully stored in: ", directory_out, sep = "")

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
    expect_error(main("data/processed", "img"))
    expect_error(main("cool/student-por", "img"))
  })
  
  test_that("The function accepts a file name as the output 
  directory when it should only accept a file path.", {
    expect_error(main("data/raw/student-por.csv", "data/processed/train.csv"))
    expect_error(main("data/raw/student-por.csv", "data/processed/coolname.csv"))
  })
  
  print("All tests have passed.")
  
}

opt <- docopt::docopt(doc)
main(opt$path_data, opt$directory_out)
tester()