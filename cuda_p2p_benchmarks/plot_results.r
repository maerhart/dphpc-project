library(ggplot2)

all_data = data.frame()

for (level in c("warp", "block", "device", "multi_device", "host_device")) {
  filename = paste("benchmark_", level, ".txt", sep="")
  df = read.csv(file=filename,head=FALSE,sep=" ")
  df = df[-nrow(df), c("V3","V7","V11")]
  colnames(df)[colnames(df) == 'V3'] <- 'Size'
  colnames(df)[colnames(df) == 'V7'] <- 'Time'
  colnames(df)[colnames(df) == 'V11'] <- 'Bandwidth'
  df[,'level'] = level
  df[,'latency'] = df[0:1, 'Time'] - 1 / tail(df, 1)['Bandwidth']
  all_data = rbind(all_data, df)
}


plot = ggplot(data=all_data, aes(x=Size, y=Bandwidth, colour=paste(level,"latency:",latency,"us")))
plot = plot + geom_line() + scale_x_log10() + xlab("Size, B") + ylab("Bandwidth, MB/s") + labs(color='level') 
plot
ggsave("bandwidths.pdf", plot = plot)


