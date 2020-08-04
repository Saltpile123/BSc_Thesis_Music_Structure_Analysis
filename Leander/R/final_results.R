setwd("C:\\help_me_pls\\_SCRIPTIE\\bachelor-scriptie-musical-sctructure-analysis\\Leander\\Results_Analysis")

cnn_1200_msaf <- read.csv("cnn_1200_msaf.csv")
#cnn_1200_own_gt <-read.csv("cnn_1200_msaf.csv") 
cnn_own_gt_msaf <- read.csv("cnn_own_gt_msaf.csv")
cnn_salami_gt_msaf <- read.csv("cnn_salami_gt_msaf.csv")


lstm_1200_msaf <- read.csv("lstm_1200_msaf.csv")
#lstm_1200_own_gt <- read.csv("cnn_1200_msaf.csv")
lstm_own_gt_msaf <- read.csv("lstm_own_gt_msaf.csv")
lstm_salami_gt_msaf <- read.csv("lstm_salami_gt_msaf.csv")

mean(lstm_salami_gt_msaf$HitRate_t3F)


