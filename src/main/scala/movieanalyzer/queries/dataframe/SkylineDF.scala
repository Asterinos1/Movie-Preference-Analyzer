package movieanalyzer.queries.dataframe

import movieanalyzer.config.AppConfig
import movieanalyzer.io.{DataBundle, ResultWriter}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{avg, col, count}

/** Query 6: Skyline Query — Non-Dominated Movies
  *
  * Identifies movies not dominated in average rating, rating count,
  * and average tag relevance using a DataFrame self-join with left_anti.
  *
  * A movie A is dominated if there exists a movie B such that B >= A in all
  * dimensions and B > A in at least one dimension. */
object SkylineDF {
  def run(spark: SparkSession, data: DataBundle, config: AppConfig): Unit = {
    val movieRatings = data.ratingsDF
      .groupBy("MovieId")
      .agg(
        avg("Rating").alias("avg_rating"),
        count("Rating").alias("rating_count")
      )

    val movieRelevance = data.genomeScoresDF
      .groupBy("MovieId")
      .agg(avg("Relevance").alias("avg_relevance"))

    val movieStats = movieRatings.join(movieRelevance, Seq("MovieId"))

    val statsA = movieStats.alias("a")
    val statsB = movieStats.alias("b")

    // Domination condition: B dominates A if B >= A in all dims AND B > A in at least one
    val dominationCondition =
      (col("b.avg_rating") >= col("a.avg_rating")) &&
        (col("b.rating_count") >= col("a.rating_count")) &&
        (col("b.avg_relevance") >= col("a.avg_relevance")) &&
        (
          (col("b.avg_rating") > col("a.avg_rating")) ||
            (col("b.rating_count") > col("a.rating_count")) ||
            (col("b.avg_relevance") > col("a.avg_relevance"))
          )

    // left_anti: keep rows from A that have NO match in B under the domination condition
    // i.e., keep movies that are NOT dominated by any other movie
    val skyline = statsA.join(statsB, dominationCondition, "left_anti")

    ResultWriter.writeDF(skyline, s"${config.outputDir}/Query6")
  }
}
