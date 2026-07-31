package movieanalyzer.queries.dataframe

import movieanalyzer.config.AppConfig
import movieanalyzer.io.{DataBundle, ResultWriter}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{avg, col}

/** Query 9: Tag-Relevance Anomaly — Overhyped Low-Rated Movies
  *
  * Identifies movies that are highly relevant (>= threshold) to popular tags
  * (e.g., "action", "classic", "thriller") but have very low average user ratings.
  */
object AnomalyDetection {
  def run(spark: SparkSession, data: DataBundle, config: AppConfig): Unit = {
    // Filter genome tags to only the specified anomaly tags
    val targetTags = data.genomeTagsDF
      .filter(col("Tag").isin(config.anomalyTags: _*))
      .select(col("TagId"))

    // Join with genome scores to get relevance for target tags only
    val highRelevanceMovies = targetTags
      .join(data.genomeScoresDF, Seq("TagId"))
      .groupBy("MovieId")
      .agg(avg("Relevance").alias("avg_relevance_to_tags"))
      .filter(col("avg_relevance_to_tags") >= config.anomalyMinRelevance)

    // Compute avg rating per movie
    val avgRatingPerMovie = data.ratingsDF
      .groupBy("MovieId")
      .agg(avg("Rating").alias("avg_rating_per_movie"))

    // Join and filter for low-rated overhyped movies
    val anomalies = highRelevanceMovies
      .join(avgRatingPerMovie, Seq("MovieId"))
      .filter(col("avg_rating_per_movie") < config.anomalyMaxRating)

    ResultWriter.writeDF(anomalies, s"${config.outputDir}/Query9")
  }
}
