library(corrplot)

mimic <- read.csv("Data/cleaned.mi (mimiciii).csv")
ami <- read.csv("Data/cleaned.mi (myocardial infarction).csv")

cor_mimic <- cor(mimic, use="pairwise.complete.obs")
cor_ami <- cor(ami, use="pairwise.complete.obs")

write.csv(cor_mimic, "Results/correlation_matrix_mimic.csv")
write.csv(cor_ami, "Results/correlation_matrix_ami.csv")

png("Results/correlation_heatmap_mimic.png", width=1200, height=1200, res=150)
corrplot(cor_mimic, method="color", type="upper", 
         tl.cex=0.5, tl.col="black",
         col=colorRampPalette(c("blue", "white", "red"))(200),
         title="MIMIC-III Correlation Matrix",
         mar=c(0,0,1,0))
dev.off()

png("Results/correlation_heatmap_ami.png", width=1600, height=1600, res=150)
corrplot(cor_ami, method="color", type="upper",
         tl.cex=0.3, tl.col="black", 
         col=colorRampPalette(c("blue", "white", "red"))(200),
         title="AMI Correlation Matrix",
         mar=c(0,0,1,0))
dev.off()
