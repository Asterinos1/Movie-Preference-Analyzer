package movieanalyzer.queries.dataframe

import movieanalyzer.config.AppConfig
import movieanalyzer.io.{DataBundle, ResultWriter}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{avg, col}

/** Query 7: Pearson Correlation
  * 
  * Computes the Pearson correlation between average tag relevance
  * and average user rating per movie.
  */
object CorrelationQuery {
  /**
    * Runs the Correlation query.
    *
    * @param spark  The SparkSession.
    * @param data   The DataBundle containing input DataFrames.
    * @param config The AppConfig containing output directories.
    */
  def run(spark: SparkSession, data: DataBundle, config: AppConfig): Unit = {
    val avgRatingPerMovieDF = data.ratingsDF
      .groupBy("MovieId")
      .agg(avg("Rating").alias("avg_rating"))
      .cache()

    val avgTagRelevancePerMovie = data.genomeScoresDF
      .groupBy("MovieId")
      .agg(avg("Relevance").alias("avg_relevance"))

    val joinedResult = avgRatingPerMovieDF.join(avgTagRelevancePerMovie, Seq("MovieId"))

    val correlation = joinedResult.stat.corr("avg_rating", "avg_relevance")

    ResultWriter.writeText(spark, s"Pearson correlation: $correlation", s"${config.outputDir}/Query7_corr")
    ResultWriter.writeDF(joinedResult, s"${config.outputDir}/Query7")
  }
}
