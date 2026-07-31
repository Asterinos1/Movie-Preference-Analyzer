package movieanalyzer.queries.rdd

import movieanalyzer.config.AppConfig
import movieanalyzer.io.{DataBundle, ResultWriter}

/** Query 2: Tag Dominance
  * 
  * For each genre, finds the most commonly used tag and computes
  * the average rating of movies with that tag.
  * 
  * Uses reduceByKey to efficiently count and find maximums without
  * large shuffles.
  */
object TagDominance {
  def run(data: DataBundle, config: AppConfig): Unit = {
    val outputPath = s"${config.outputDir}/Query2"
    
    val movieTagPairs = data.tagsRDD.map { case (_, movieId, tag) => (movieId, tag) }
    
    val movieGenrePairs = data.moviesRDD.flatMap { case (movieId, _, genres) =>
      genres.filter(_ != "(no genres listed)").map(genre => (movieId, genre))
    }
    
    // Join tags with genres on movieId
    val tagGenreJoined = movieTagPairs.join(movieGenrePairs) // (movieId, (tag, genre))
    tagGenreJoined.cache()
    
    // 1. Count occurrences of each (genre, tag) pair
    val genreTagCounts = tagGenreJoined
      .map { case (_, (tag, genre)) => ((genre, tag), 1) }
      .reduceByKey(_ + _)
    
    // 2. Find most used tag per genre
    val mostUsedTagPerGenre = genreTagCounts
      .map { case ((genre, tag), count) => (genre, (tag, count)) }
      .reduceByKey((a, b) => if (a._2 > b._2) a else b)
    
    // 3. Extract the dominant tags: Set of (genre, tag)
    val dominantTagGenresMap = mostUsedTagPerGenre
      .map { case (genre, (tag, _)) => ((genre, tag), true) }
      .collectAsMap()
    
    val broadcastDominantTags = data.moviesRDD.sparkContext.broadcast(dominantTagGenresMap)
    
    // 4. Filter original tagGenreJoined to keep only movies with dominant tag
    val dominantMovies = tagGenreJoined.filter { case (_, (tag, genre)) =>
      broadcastDominantTags.value.contains((genre, tag))
    }
    
    // 5. Join with avgRatingPerMovie, compute avg rating per (genre, tag)
    val avgRatingPerMovie = data.ratingsRDD
      .map { case (_, movieId, rating) => (movieId, (rating, 1)) }
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .mapValues { case (sum, count) => sum / count }
    
    val dominantWithRatings = dominantMovies.join(avgRatingPerMovie)
      .map { case (_, ((tag, genre), avgRating)) => ((genre, tag), avgRating) }
    
    val result = dominantWithRatings
      .aggregateByKey((0.0, 0))(
        (acc, rating) => (acc._1 + rating, acc._2 + 1),
        (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2)
      )
      .mapValues { case (sum, count) => sum / count }
      
    ResultWriter.writeRDD(result, outputPath)
  }
}
