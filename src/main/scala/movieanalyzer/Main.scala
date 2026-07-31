package movieanalyzer

import movieanalyzer.config.{AppConfig, SparkSessionFactory}
import movieanalyzer.io.{DataBundle, DataLoader}
import movieanalyzer.queries.QueryRunner

object Main {
  def main(args: Array[String]): Unit = {
    val config = AppConfig.load(args)
    val spark = SparkSessionFactory.create(config)
    
    try {
      println(s"Movie Preference Analyzer starting with profile: ${config.sparkMaster}")
      println(s"Data directory: ${config.dataDir}")
      println(s"Output directory: ${config.outputDir}")
      
      val data = DataLoader.load(spark, config.dataDir)
      val selectedQueries = parseQuerySelection(args)
      
      QueryRunner.run(spark, data, config, selectedQueries)
      
      println("All queries completed successfully.")
    } finally {
      spark.stop()
    }
  }
  
  private def parseQuerySelection(args: Array[String]): Set[Int] = {
    val idx = args.indexWhere(a => a == "--query" || a.startsWith("--query="))
    if (idx < 0) return (1 to 10).toSet
    
    val value = if (args(idx).contains("=")) args(idx).split("=", 2)(1)
                else if (idx + 1 < args.length) args(idx + 1)
                else "all"
    
    if (value == "all") (1 to 10).toSet
    else value.split(",").map(_.trim.toInt).toSet
  }
}
