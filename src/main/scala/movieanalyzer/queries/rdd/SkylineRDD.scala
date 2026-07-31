package movieanalyzer.queries.rdd

import movieanalyzer.config.AppConfig
import movieanalyzer.io.{DataBundle, ResultWriter}

/** Query 5: Skyline Analysis
  *
  * Multi-Iceberg Skyline over genre-tag-user triads.
  *
  * PERFORMANCE FIX: Replaces Cartesian self-join with a local O(N log N)
  * skyline algorithm after collecting the reduced dataset, which easily fits in memory.
  *
  * The skyline is computed purely via foldLeft — no mutable state. */
object SkylineRDD {
  def run(data: DataBundle, config: AppConfig): Unit = {
    val minMovies = config.skylineMinMovies
    val outputPath = s"${config.outputDir}/Query5"

    val movieTagPairs = data.tagsRDD.map { case (userId, movieId, tag) => (movieId, (userId, tag)) }

    val movieGenrePairs = data.moviesRDD.flatMap { case (movieId, _, genres) =>
      genres.filter(_ != "(no genres listed)").map(genre => (movieId, genre))
    }

    // Join tags with genres on movieId: (movieId, ((userId, tag), genre))
    val tagGenreJoined = movieTagPairs.join(movieGenrePairs)
    tagGenreJoined.cache()

    // Count genre-tag pair occurrences, keep only those > minMovies
    val counts = tagGenreJoined
      .map { case (_, ((_, tag), genre)) => ((genre, tag), 1) }
      .reduceByKey(_ + _)
      .filter(_._2 > minMovies)

    val validPairs = counts.keys.collect().toSet
    val broadcastValidPairs = data.moviesRDD.sparkContext.broadcast(validPairs)

    val filteredTagGenre = tagGenreJoined.filter { case (_, ((_, tag), genre)) =>
      broadcastValidPairs.value.contains((genre, tag))
    }
    filteredTagGenre.cache()

    // Compute unique user count per (genre, tag)
    // Using distinct + reduceByKey instead of groupByKey
    val userCounts = filteredTagGenre
      .map { case (_, ((userId, tag), genre)) => ((genre, tag), userId) }
      .distinct()
      .map { case (genreTag, _) => (genreTag, 1) }
      .reduceByKey(_ + _)

    // Compute avg rating per (genre, tag) via reduceByKey
    val avgRatingPerMovie = data.ratingsRDD
      .map { case (_, movieId, rating) => (movieId, (rating, 1)) }
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .mapValues { case (sum, count) => sum / count }

    val movieAvgRatings = filteredTagGenre
      .map { case (movieId, ((_, tag), genre)) => (movieId, (genre, tag)) }
      .distinct()
      .join(avgRatingPerMovie)
      .map { case (_, ((genre, tag), avgRating)) => ((genre, tag), (avgRating, 1)) }
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .mapValues { case (sum, count) => sum / count }

    // Join user counts and avg ratings -> ((genre, tag), (userCount, avgRating))
    val combinedStats = userCounts.join(movieAvgRatings).collect()

    // FUNCTIONAL skyline computation via foldLeft — no var, no mutable collection
    // Sort by first dimension (userCount) descending
    val sorted = combinedStats.sortBy { case (_, (uCount, _)) => -uCount }

    // foldLeft accumulates (skylinePoints, maxRatingSeenSoFar)
    // A point enters the skyline if its rating exceeds the max rating seen so far
    // (since userCount is already sorted desc, this correctly identifies non-dominated points)
    val (skylinePoints, _) = sorted.foldLeft((List.empty[((String, String), (Int, Double))], -1.0)) {
      case ((acc, maxRating), item) =>
        val rating = item._2._2
        if (rating > maxRating) (item :: acc, rating)
        else (acc, maxRating)
    }

    val resultRDD = data.moviesRDD.sparkContext.parallelize(skylinePoints.reverse)

    ResultWriter.writeRDD(resultRDD, outputPath)
  }
}
