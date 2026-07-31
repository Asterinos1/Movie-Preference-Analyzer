package movieanalyzer.queries.rdd

import movieanalyzer.config.AppConfig
import movieanalyzer.io.{DataBundle, ResultWriter}

/** Query 3: Popular Tags
  * 
  * Finds tags appearing in > N unique movies with average relevance > threshold.
  * 
  * PERFORMANCE FIX: Replaces groupByKey with aggregateByKey.
  * Accumulates (Set[movieId], sumRelevance, count) only for relevance > minRelevance.
  */
object PopularTags {
  def run(data: DataBundle, config: AppConfig): Unit = {
    val minRelevance = config.popularTagsMinRelevance
    val minMovies = config.popularTagsMinMovies
    val minAvgRelevance = config.popularTagsMinAvgRelevance
    val outputPath = s"${config.outputDir}/Query3"
    
    // PERFORMANCE FIX: Use aggregateByKey to compute unique movies and relevance sums without groupByKey
    val tagStats = data.genomeScoresRDD
      .filter { case (_, _, relevance) => relevance > minRelevance }
      .map { case (movieId, tagId, relevance) => (tagId, (movieId, relevance)) }
      .aggregateByKey((Set.empty[String], 0.0, 0))(
        (acc, item) => (acc._1 + item._1, acc._2 + item._2, acc._3 + 1),
        (acc1, acc2) => (acc1._1 ++ acc2._1, acc1._2 + acc2._2, acc1._3 + acc2._3)
      )
      
    val filteredTags = tagStats
      .filter { case (_, (movieSet, _, _)) => movieSet.size > minMovies }
      .mapValues { case (_, sumRelevance, count) => sumRelevance / count }
      .filter { case (_, avgRelevance) => avgRelevance > minAvgRelevance }
      
    // Join with genomeTagsRDD to get tag names
    val result = filteredTags
      .join(data.genomeTagsRDD)
      .map { case (tagId, (avgRelevance, tagName)) => (tagId, tagName, avgRelevance) }
      
    ResultWriter.writeRDD(result, outputPath)
  }
}
