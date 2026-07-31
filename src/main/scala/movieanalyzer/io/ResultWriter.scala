package movieanalyzer.io

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

/** Unified result writer supporting RDD and DataFrame output formats. */
object ResultWriter {
  def writeRDD[T](rdd: RDD[T], path: String): Unit = {
    val hadoopConf = rdd.sparkContext.hadoopConfiguration
    val fs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val outputPath = new org.apache.hadoop.fs.Path(path)
    if (fs.exists(outputPath)) fs.delete(outputPath, true)
    rdd.saveAsTextFile(path)
  }
  
  def writeDF(df: DataFrame, path: String): Unit = {
    df.write.mode("overwrite").option("header", value = true).csv(path)
  }
  
  def writeText(spark: SparkSession, text: String, path: String): Unit = {
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val fs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val outputPath = new org.apache.hadoop.fs.Path(path)
    if (fs.exists(outputPath)) fs.delete(outputPath, true)
    spark.sparkContext.parallelize(Seq(text)).coalesce(1).saveAsTextFile(path)
  }
}
