/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.evaluation

import org.apache.spark.mllib.evaluation.binary._
import org.apache.spark.rdd.RDD

/**
 * Evaluator for binary classification.
 *
 * @param scoreAndLabels an RDD of (score, label) pairs.
 * @param numBins if greater than 0, then the curves (ROC curve, PR curve) computed internally
 *                will be down-sampled to this many "bins". If 0, no down-sampling will occur.
 *                This is useful because the curve contains a point for each distinct score
 *                in the input, and this could be as large as the input itself -- millions of
 *                points or more, when thousands may be entirely sufficient to summarize
 *                the curve. After down-sampling, the curves will instead be made of approximately
 *                `numBins` points instead. Points are made from bins of equal numbers of
 *                consecutive points. The size of each bin is
 *                `floor(scoreAndLabels.count() / numBins)`, which means the resulting number
 *                of bins may not exactly equal numBins. The last bin in each partition may
 *                be smaller as a result, meaning there may be an extra sample at
 *                partition boundaries.
 */
class RichBinaryClassificationMetrics(
  override val scoreAndLabels: RDD[(Double, Double)],
  override val numBins: Int
) extends BinaryClassificationMetrics(scoreAndLabels, numBins) {

  /**
   * Returns the confusion matrix at each threshold.
   */
  def confusionMatrixByThreshold(): RDD[(Double, BinaryConfusionMatrix)] = confusions

  private[mllib] lazy val confusions: RDD[(Double, BinaryConfusionMatrix)] = {
    // Create a bin for each distinct score value, count positives and negatives within each bin,
    // and then sort by score values in descending order.
    val counts = scoreAndLabels.combineByKey(
      createCombiner = (label: Double) => new BinaryLabelCounter(0L, 0L) += label,
      mergeValue = (c: BinaryLabelCounter, label: Double) => c += label,
      mergeCombiners = (c1: BinaryLabelCounter, c2: BinaryLabelCounter) => c1 += c2
    ).sortByKey(ascending = false)

    val binnedCounts =
    // Only down-sample if bins is > 0
      if (numBins == 0) {
        // Use original directly
        counts
      } else {
        val countsSize = counts.count()
        // Group the iterator into chunks of about countsSize / numBins points,
        // so that the resulting number of bins is about numBins
        var grouping = countsSize / numBins
        if (grouping < 2) {
          // numBins was more than half of the size; no real point in down-sampling to bins
          logInfo(s"Curve is too small ($countsSize) for $numBins bins to be useful")
          counts
        } else {
          if (grouping >= Int.MaxValue) {
            logWarning(
              s"Curve too large ($countsSize) for $numBins bins; capping at ${Int.MaxValue}")
            grouping = Int.MaxValue
          }
          counts.mapPartitions(_.grouped(grouping.toInt).map { pairs =>
            // The score of the combined point will be just the last one's score, which is also
            // the minimal in each chunk since all scores are already sorted in descending.
            val lastScore = pairs.last._1
            // The combined point will contain all counts in this chunk. Thus, calculated
            // metrics (like precision, recall, etc.) on its score (or so-called threshold) are
            // the same as those without sampling.
            val agg = new BinaryLabelCounter()
            pairs.foreach(pair => agg += pair._2)
            (lastScore, agg)
          })
        }
      }

    val agg = binnedCounts.values.mapPartitions { iter =>
      val agg = new BinaryLabelCounter()
      iter.foreach(agg += _)
      Iterator(agg)
    }.collect()
    val partitionwiseCumulativeCounts =
      agg.scanLeft(new BinaryLabelCounter())((agg, c) => agg.clone() += c)
    val totalCount = partitionwiseCumulativeCounts.last
    logInfo(s"Total counts: $totalCount")
    val cumulativeCounts = binnedCounts.mapPartitionsWithIndex(
      (index: Int, iter: Iterator[(Double, BinaryLabelCounter)]) => {
        val cumCount = partitionwiseCumulativeCounts(index)
        iter.map { case (score, c) =>
          cumCount += c
          (score, cumCount.clone())
        }
      }, preservesPartitioning = true)
    cumulativeCounts.persist()
    cumulativeCounts.map { case (score, cumCount) =>
      (score, BinaryConfusionMatrixImpl(cumCount, totalCount).asInstanceOf[BinaryConfusionMatrix])
    }
  }
}
