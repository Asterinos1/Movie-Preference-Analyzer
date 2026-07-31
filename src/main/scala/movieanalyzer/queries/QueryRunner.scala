package movieanalyzer.queries

import movieanalyzer.config.AppConfig
import movieanalyzer.io.DataBundle
import movieanalyzer.queries.rdd._
import movieanalyzer.queries.dataframe._
import org.apache.spark.sql.{DataFrame, SparkSession}

/** Orchestrates running selected queries with timing and dependency management.
  *
  * Uses foldLeft to thread Q8's cosine DataFrame through the query sequence
  * without mutable state. */
object QueryRunner {

  /** Runs the selected queries in order, managing cross-query dependencies functionally.
    *
    * @param spark           Active SparkSession
    * @param data            Loaded DataBundle
    * @param config          Application configuration
    * @param selectedQueries Set of query numbers to run (1-10)
    */
  def run(spark: SparkSession, data: DataBundle, config: AppConfig, selectedQueries: Set[Int]): Unit = {
    // foldLeft threads the optional cosine DF (Q8 output) through the query sequence.
    // This avoids a `var` — the accumulator carries state between iterations.
    val finalCosineDF = selectedQueries.toSeq.sorted.foldLeft(Option.empty[DataFrame]) {
      case (cosineDF, qNum) =>
        println(s"\n${"-" * 60}")
        println(s"Running Query $qNum: ${queryName(qNum)}")
        println(s"${"-" * 60}")
        val startTime = System.currentTimeMillis()

        val nextCosineDF = qNum match {
          case 1  => IcebergQuery.run(data, config); cosineDF
          case 2  => TagDominance.run(data, config); cosineDF
          case 3  => PopularTags.run(data, config); cosineDF
          case 4  => SentimentEstimation.run(data, config); cosineDF
          case 5  => SkylineRDD.run(data, config); cosineDF
          case 6  => SkylineDF.run(spark, data, config); cosineDF
          case 7  => CorrelationQuery.run(spark, data, config); cosineDF
          case 8  =>
            val df = ReverseNNQuery.run(spark, data, config)
            Some(df)
          case 9  => AnomalyDetection.run(spark, data, config); cosineDF
          case 10 =>
            TopKNeighborhood.run(spark, data, config, cosineDF)
            cosineDF.foreach(_.unpersist())
            None
          case _  =>
            println(s"Warning: Query $qNum not found. Valid range: 1-10.")
            cosineDF
        }

        val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
        println(f"Query $qNum completed in $elapsed%.1f seconds.")
        nextCosineDF
    }

    // Cleanup if Q8 ran but Q10 didn't
    finalCosineDF.foreach(_.unpersist())
  }

  private def queryName(q: Int): String = q match {
    case 1  => "Iceberg Query — Top Tags by Genre"
    case 2  => "Tag Dominance per Genre"
    case 3  => "Popular and Relevant Tags"
    case 4  => "Sentiment Estimation"
    case 5  => "Multi-Iceberg Skyline"
    case 6  => "Skyline — Non-Dominated Movies"
    case 7  => "Correlation: Relevance vs Ratings"
    case 8  => "Reverse Nearest Neighbor"
    case 9  => "Tag-Relevance Anomaly"
    case 10 => "Reverse Top-K Neighborhood"
    case _  => "Unknown"
  }
}
