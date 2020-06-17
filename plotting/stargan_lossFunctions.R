### Script for plotting
library(dplyr)
library(reshape2)
library(ggplot2)

data1 <- read.csv("~/PycharmProjects/logs/stargan/spraakbanken/25_Spraakbanken-Test_log.txt", header=FALSE)

data <- read.csv("~/PycharmProjects/logs/stargan/spraakbanken/25_Spraakbanken-Test_log2.txt", header=FALSE)

# Rename columns
colnames(data)
data <- select(data, -c("V8"))
names(data)[names(data) == "V1"] <- "D/loss_real"
names(data)[names(data) == "V2"] <- "D/loss_fake"
names(data)[names(data) == "V3"] <- "D/loss_cls_spks"
names(data)[names(data) == "V4"] <- "D/loss_gp"
names(data)[names(data) == "V5"] <- "G/loss_fake"
names(data)[names(data) == "V6"] <- "G/loss_rec"
names(data)[names(data) == "V7"] <- " G/loss_cls_spks"

# Getting long data format
data2 <- melt(data)
data2$rowid <- 1:20000
ggplot(data2, aes(variable, value, group=factor(rowid))) + geom_point(aes(color=factor(rowid)))


# Now we do plotting
plot(data)  # Correlation plot
summary(data)

plot(c(1:20000),data$`D/loss_real`)
plot(c(1:20000),data$`D/loss_fake`)

