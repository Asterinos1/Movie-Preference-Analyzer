package movieanalyzer.queries.rdd

import movieanalyzer.config.AppConfig
import movieanalyzer.io.{DataBundle, ResultWriter}

/** Query 1: Iceberg Query — Top Tags by Genre with High Ratings
  * 
  * Finds (genre, tag) pairs that appear in more than N movies
  * and have an average user rating greater than a threshold.
  * 
  * Uses aggregateByKey instead of groupByKey for O(1) memory per partition.
  */
object IcebergQuery {
  def run(data: DataBundle, config: AppConfig): Unit = {
    val minMovies = config.icebergMinMovies
    val minAvgRating = config.icebergMinAvgRating
    val outputPath = s"${config.outputDir}/Query1"
    
    // Compute avg rating per movie
    val avgRatingPerMovie = data.ratingsRDD
      .map { case (_, movieId, rating) => (movieId, (rating, 1)) }
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .mapValues { case (sum, count) => sum / count }
    
    val movieTagPairs = data.tagsRDD.map { case (_, movieId, tag) => (movieId, tag) }
    
    val movieGenrePairs = data.moviesRDD.flatMap { case (movieId, _, genres) =>
      genres.filter(_ != "(no genres listed)").map(genre => (movieId, genre))
    }
    
    // Join tags with genres on movieId
    val tagGenreJoined = movieTagPairs.join(movieGenrePairs) // (movieId, (tag, genre))
    
    // Join with avg ratings
    val withRatings = tagGenreJoined.join(avgRatingPerMovie)
      .map { case (_, ((tag, genre), avgRating)) => ((tag, genre), avgRating) }
    
    // PERFORMANCE FIX: Use aggregateByKey instead of groupByKey
    // Accumulator: (sumOfRatings, count)
    // Avoids shuffling all ratings into a list per key, keeping memory usage constant.
    val result = withRatings
      .aggregateByKey((0.0, 0))(
        (acc, rating) => (acc._1 + rating, acc._2 + 1),
        (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2)
      )
      .filter { case (_, (_, count)) => count > minMovies }
      .mapValues { case (sum, count) => sum / count }
      .filter { case (_, avg) => avg > minAvgRating }
    
    ResultWriter.writeRDD(result, outputPath)
  }
}
