library(ggplot2)

all_data = data.frame()

for (level in c("warp", "block", "device", "multi_device", "host_device")) {
  filename = paste("benchmark_", level, ".txt", sep="")
  df = read.csv(file=filename,head=FALSE,sep=" ")
  df = df[-nrow(df), c("V3","V7","V10","V15","V18")]
  colnames(df)[colnames(df) == 'V3'] <- 'Size'
  colnames(df)[colnames(df) == 'V7'] <- 'Time'
  colnames(df)[colnames(df) == 'V10'] <- 'TimeErr'
  colnames(df)[colnames(df) == 'V15'] <- 'Bandwidth'
  colnames(df)[colnames(df) == 'V18'] <- 'BandwidthErr'
  df[,'level'] = level
  df[,'latency'] = df[1, 'Time'] - df[2, 'Time'] / 2
  df[,'latencyErr'] = df[1, 'TimeErr'] + df[2, 'TimeErr'] / 2
  all_data = rbind(all_data, df)
}

all_data[1,]

plot = ggplot(data=all_data, aes(x=Size, y=Bandwidth, ymin=Bandwidth - BandwidthErr, ymax=Bandwidth + BandwidthErr,
                                 colour=sprintf("%s latency: %.2f ( +- %.2f ) us",level,latency,latencyErr)))
plot = plot + geom_line() + xlab("Size, B") + ylab("Bandwidth, MB/s") + labs(color='level') 
plot = plot + geom_ribbon(alpha=0.5)
plot = plot + scale_x_log10()
plot = plot + ggtitle(label="Bandwidth and latency for different levels of CUDA communication", 
                      subtitle = "Plot shows performance for *ideal* scenario with *measurement* errors, but not possible *divergence* in practise")
plot

ggsave("bandwidths.pdf", plot = plot)

# The following code was used to detect source of high variance in data (for multi-device and host-device case).
# After ~20 iterations high peaks in latencies stop to appear, so
# benchmark is fixed not to measure first 20 iterations. 
# IMPORTANT: obtained results are for "ideal" case, in practise 
# latencies can be much higher and (+- values) in this plot denote 
# errors of measurement of "ideal" case, but not variance in practise. 

require(reshape2)

df = read.csv(file="host_device_raw_data.txt",head=FALSE,sep=" ")
df = melt(df, id.vars = c("V1"))
df$dataSize <- df$V1
df$variable = NULL
df$V1 = NULL

df1 = df[df$dataSize < 2 ** 10,]

ggplot(data=df1, aes(x=as.factor(dataSize), y=value)) +
  geom_boxplot() +
  scale_y_log10() +
  xlab("Size, B") +
  ylab("Time, ms") +
  ggtitle(label="Host-device time measurement box plot", 
          subtitle = "Outliers are mostly from the first 20 measurements (out of 120), boxes are too thin: measurement is precise for last 100")

ggsave("host_device_box_plot.pdf")
