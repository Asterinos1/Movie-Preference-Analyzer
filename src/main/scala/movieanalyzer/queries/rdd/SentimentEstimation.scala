package movieanalyzer.queries.rdd

import movieanalyzer.config.AppConfig
import movieanalyzer.io.{DataBundle, ResultWriter}

/** Query 4: Sentiment Estimation
  * 
  * Computes the average user rating for each user-assigned tag.
  * 
  * Efficiently uses join and reduceByKey to calculate averages.
  */
object SentimentEstimation {
  def run(data: DataBundle, config: AppConfig): Unit = {
    val outputPath = s"${config.outputDir}/Query4"
    
    // tagByUserMovie = tagsRDD.map((userId, movieId) -> tag)
    val tagByUserMovie = data.tagsRDD.map { case (userId, movieId, tag) => 
      ((userId, movieId), tag) 
    }
    
    // ratingByUserMovie = ratingsRDD.map((userId, movieId) -> rating)
    val ratingByUserMovie = data.ratingsRDD.map { case (userId, movieId, rating) => 
      ((userId, movieId), rating) 
    }
    
    // Join and compute average rating per tag
    val result = tagByUserMovie.join(ratingByUserMovie)
      .map { case (_, (tag, rating)) => (tag, (rating, 1)) }
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .mapValues { case (sum, count) => sum / count }
      
    ResultWriter.writeRDD(result, outputPath)
  }
}
