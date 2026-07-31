package movieanalyzer.queries.dataframe

import movieanalyzer.config.AppConfig
import movieanalyzer.io.{DataBundle, ResultWriter}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number}

/** Query 10: Top-K Neighborhood
  * 
  * Finds the Top-K users closest to a target movie using Cosine Similarity.
  */
object TopKNeighborhood {
  /**
    * Runs the Top-K Neighborhood query.
    *
    * @param spark    The SparkSession.
    * @param data     The DataBundle containing input DataFrames.
    * @param config   The AppConfig containing Top-K configuration.
    * @param cosineDF Optional DataFrame containing persisted cosine similarities from Query 8.
    */
  def run(spark: SparkSession, data: DataBundle, config: AppConfig, cosineDF: Option[DataFrame] = None): Unit = {
    val baseDF = cosineDF.getOrElse(ReverseNNQuery.run(spark, data, config))

    val windowSpec = Window.orderBy(col("cosine_similarity").desc)

    val topK = baseDF
      .withColumn("Rank", row_number().over(windowSpec))
      .filter(col("Rank") <= config.topK)

    ResultWriter.writeDF(topK, s"${config.outputDir}/Query10")
  }
}
