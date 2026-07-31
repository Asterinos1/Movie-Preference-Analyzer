package movieanalyzer.queries.dataframe

import movieanalyzer.config.AppConfig
import movieanalyzer.io.{DataBundle, ResultWriter}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{avg, col, sum, sqrt}

/** Query 8: Reverse Nearest Neighbor
  * 
  * Matches users to a target movie via Cosine Similarity.
  */
object ReverseNNQuery {
  /**
    * Runs the Reverse NN query and returns the cosine components DataFrame.
    *
    * @param spark  The SparkSession.
    * @param data   The DataBundle containing input DataFrames.
    * @param config The AppConfig containing thresholds and target movie ID.
    * @return The persisted cosine components DataFrame.
    */
  def run(spark: SparkSession, data: DataBundle, config: AppConfig): DataFrame = {
    val targetMovieProfile = data.genomeScoresDF
      .filter(col("MovieId") === config.reverseNnTargetMovieId.toString)
      .select(col("TagId"), col("Relevance").alias("target_score"))

    val likedMovies = data.ratingsDF
      .filter(col("Rating") > config.reverseNnLikedThreshold)

    val userProfiles = likedMovies
      .join(data.genomeScoresDF, Seq("MovieId"))
      .groupBy("UserId", "TagId")
      .agg(avg("Relevance").alias("user_score"))

    val tagProfilesJoined = userProfiles.join(targetMovieProfile, Seq("TagId"))

    val cosineComponentsDF = tagProfilesJoined
      .withColumn("dot", col("user_score") * col("target_score"))
      .withColumn("user_norm_sqr", col("user_score") * col("user_score"))
      .withColumn("target_norm_sqr", col("target_score") * col("target_score"))
      .groupBy("UserId")
      .agg(
        sum("dot").alias("dot_product"),
        sum("user_norm_sqr").alias("user_norm_sqr"),
        sum("target_norm_sqr").alias("target_norm_sqr")
      )
      .withColumn("cosine_similarity",
        col("dot_product") / (sqrt(col("user_norm_sqr")) * sqrt(col("target_norm_sqr"))))
      .select("UserId", "cosine_similarity")
      .persist()

    ResultWriter.writeDF(cosineComponentsDF, s"${config.outputDir}/Query8")
    
    cosineComponentsDF
  }
}
