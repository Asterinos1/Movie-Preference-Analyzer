package movieanalyzer.config

import org.apache.spark.sql.SparkSession

/** Factory for creating SparkSession instances configured per deployment profile. */
object SparkSessionFactory {
  def create(config: AppConfig): SparkSession = {
    val builder = SparkSession.builder
      .appName(config.appName)
      .master(config.sparkMaster)
    
    // If cluster mode, set hadoop configs
    config.hadoopFsDefault.foreach(uri => builder.config("spark.hadoop.fs.defaultFS", uri))
    config.hadoopYarnRm.foreach(rm => builder.config("spark.hadoop.yarn.resourcemanager.address", rm))
    config.hadoopYarnClasspath.foreach(cp => builder.config("spark.hadoop.yarn.application.classpath", cp))
    
    builder.getOrCreate()
  }
}
