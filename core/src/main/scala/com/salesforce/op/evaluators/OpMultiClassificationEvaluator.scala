/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.evaluators

import com.fasterxml.jackson.databind.annotation.JsonDeserialize
import com.salesforce.op.UID
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.Tuple2Semigroup
import com.salesforce.op.utils.spark.RichEvaluator._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector, DenseVector}
import org.apache.spark.ml.param.{DoubleArrayParam, IntArrayParam, IntParam, ParamValidators}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}
import org.slf4j.LoggerFactory
import scala.collection.Searching._


/**
 * Instance to evaluate Multi Classification metrics
 * The metrics are  Precision, Recall, F1 and Error Rate
 * Default evaluation returns F1 score
 *
 * @param name           name of default metric
 * @param uid            uid for instance
 */
private[op] class OpMultiClassificationEvaluator
(
  override val name: EvalMetric = OpEvaluatorNames.Multi,
  override val uid: String = UID[OpMultiClassificationEvaluator]
) extends OpMultiClassificationEvaluatorBase[MultiClassificationMetrics](uid) {

  private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: MultiClassificationMetrics => Double = _.F1

  final val topNs = new IntArrayParam(
    parent = this,
    name = "topNs",
    doc = "sequence of topN values to use for threshold metrics",
    isValid = _.forall(_ > 0)
  )
  setDefault(topNs, Array(1, 3))

  def setTopNs(v: Array[Int]): this.type = set(topNs, v)

  final val topKs = new IntArrayParam(
    parent = this,
    name = "topKs",
    doc = "sequence of topK values to use for top K metrics",
    isValid = l => l.length <= 10 && l.forall(_ > 0)
  )
  setDefault(topKs, Array(5, 10, 20, 50, 100))

  def setTopKs(v: Array[Int]): this.type = set(topKs, v)

  final val thresholds = new DoubleArrayParam(
    parent = this,
    name = "thresholds",
    doc = "sequence of threshold values (must be in [0.0, 1.0]) to use for threshold metrics",
    isValid = _.forall(x => x >= 0.0 && x <= 1.0)
  )
  setDefault(thresholds, (0 to 100).map(_ / 100.0).toArray)

  def setThresholds(v: Array[Double]): this.type = set(thresholds, v)

  final val confMatrixNumClasses = new IntParam(
    parent = this,
    name = "confMatrixNumClasses",
    doc = "# of the top most frequent classes used for confusion matrix metrics",
    isValid = ParamValidators.inRange(1, 30, lowerInclusive = true, upperInclusive = true)
  )
  setDefault(confMatrixNumClasses, 15)

  def setConfMatrixNumClasses(v: Int): this.type = set(confMatrixNumClasses, v)

  final val confMatrixMinSupport = new IntParam(
    parent = this,
    name = "confMatrixMinSupport",
    doc = "# of the top most frequent misclassified classes in each label/prediction category",
    isValid = ParamValidators.inRange(1, 10, lowerInclusive = false, upperInclusive = true)
  )
  setDefault(confMatrixMinSupport, 5)

  def setConfMatrixMinSupport(v: Int): this.type = set(confMatrixMinSupport, v)

  final val confMatrixThresholds = new DoubleArrayParam(
    parent = this,
    name = "confMatrixThresholds",
    doc = "sequence of threshold values used for confusion matrix metrics",
    isValid = _.forall(x => x >= 0.0 && x < 1.0)
  )
  setDefault(confMatrixThresholds, Array(0.0, 0.2, 0.4, 0.6, 0.8))
  def setConfMatrixThresholds(v: Array[Double]): this.type = set(confMatrixThresholds, v)

  override def evaluateAll(data: Dataset[_]): MultiClassificationMetrics = {
    val labelColName = getLabelCol
    val dataUse = makeDataToUse(data, labelColName)

    val (predictionColName, rawPredictionColName, probabilityColName) = (getPredictionValueCol,
      getRawPredictionCol, getProbabilityCol)

    log.debug(
      "Evaluating metrics on columns :\n label : {}\n rawPrediction : {}\n prediction : {}\n probability : {}\n",
      labelColName, rawPredictionColName, predictionColName, probabilityColName
    )

    import dataUse.sparkSession.implicits._
    val rdd = dataUse.select(predictionColName, labelColName).as[(Double, Double)].rdd
    if (rdd.isEmpty()) {
      log.warn("The dataset is empty. Returning empty metrics.")
      MultiClassificationMetrics(0.0, 0.0, 0.0, 0.0,
        MulticlassThresholdMetrics(Seq.empty, Seq.empty, Map.empty, Map.empty, Map.empty),
        MultiClassificationMetricsTopK(Seq.empty, Seq.empty, Seq.empty, Seq.empty, Seq.empty),
        MulticlassConfMatrixMetricsByThreshold($(confMatrixNumClasses), Seq.empty, $(confMatrixThresholds), Seq.empty),
        MisClassificationMetrics($(confMatrixMinSupport), Seq.empty, Seq.empty))
    } else {
      val multiclassMetrics = new MulticlassMetrics(rdd)
      val error = 1.0 - multiclassMetrics.accuracy
      val precision = multiclassMetrics.weightedPrecision
      val recall = multiclassMetrics.weightedRecall
      val f1 = if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)

      val thresholdMetrics = calculateThresholdMetrics(
        data = dataUse.select(col(probabilityColName), col(labelColName).cast(DoubleType)).rdd.map{
          case Row(prob: Vector, label: Double) => (prob.toArray, label)
        },
        topNs = $(topNs),
        thresholds = $(thresholds)
      )

      val topKMetrics = calculateTopKMetrics(
        data = rdd,
        topKs = $(topKs)
      )

      val rddCm = dataUse.select(col(labelColName), col(predictionColName), col(probabilityColName)).rdd.map{
        case Row(label: Double, pred: Double, prob: DenseVector) => (label, pred, prob.toArray)
      }
      val confusionMatrixByThreshold = calculateConfMatrixMetricsByThreshold(rddCm)
      val misClassifications = calculateMisClassificationMetrics( rddCm.map{ case (label, pred, _) => (label, pred)} )

      val metrics = MultiClassificationMetrics(
        Precision = precision,
        Recall = recall,
        F1 = f1,
        Error = error,
        ThresholdMetrics = thresholdMetrics,
        TopKMetrics = topKMetrics,
        ConfusionMatrixMetrics = confusionMatrixByThreshold,
        MisClassificationMetrics = misClassifications
      )

      log.info("Evaluated metrics: {}", metrics.toString)
      metrics
    }
  }

  case class LabelPredictionConfidenceCt(Label: Double, Prediction: Double, Confidence: Double, count: Long)

/**
 * function to construct the confusion matrix for the top n most occurring labels
 * @param labelPredictionCtRDD RDD of ((label, prediction, confidence), count)
 * @param cmClasses the top n most occurring classes, sorted by counts in descending order
 * @return an array of counts
 */
  def constructConfusionMatrix(
    labelPredictionCtRDD: RDD[((Double, Double, Double), Long)],
    cmClasses: Seq[Double]): Seq[Long] = {

    val confusionMatrixMap = labelPredictionCtRDD.map {
      case ((label, prediction, _), count) => ((label, prediction), count)
    }.reduceByKey(_ + _).collectAsMap()

    for {
      label <- cmClasses
      prediction <- cmClasses
    } yield {
      confusionMatrixMap.getOrElse((label, prediction), 0L)
    }
  }

  private[evaluators] object SearchHelper extends Serializable{

    /**
     * function to search the confidence threshold corresponding to a probability score
     *
     * @param arr a sorted array of confidence thresholds
     * @param element the probability score to be searched
     * @return the confidence threshold corresponding of the element. It equals to the element if there is an exact
     *         match. Otherwise it's the element right before the insertion point.
     */
    def findThreshold(arr: IndexedSeq[Double], element: Double): Double = {
      require(!arr.isEmpty, "Array of confidence thresholds can't be empty!")
      if (element > arr.last) arr.last
      else if (element < arr.head) 0.0
      else {
        val insertionPoint = new SearchImpl(arr).search(element).insertionPoint
        val insertionPointValue = arr(insertionPoint)
        if (element == insertionPointValue) insertionPointValue
        else arr(insertionPoint-1)
      }
    }
  }

/**
 * function to calculate confusion matrix for TopK most occurring labels by confidence threshold
 *
 * @param data RDD of (label, prediction, prediction probability vector)
 * @return a MulticlassConfMatrixMetricsByThreshold instance
*/
  def calculateConfMatrixMetricsByThreshold(
    data: RDD[(Double, Double, Array[Double])]): MulticlassConfMatrixMetricsByThreshold = {

    val labelCountsRDD = data.map { case (label, _, _) => (label, 1L) }.reduceByKey(_ + _)
    val cmClasses = labelCountsRDD.sortBy(-_._2).map(_._1).take($(confMatrixNumClasses)).toSeq
    val cmClassesSet = cmClasses.toSet

    val dataTopNLabels = data.filter { case (label, prediction, _) =>
      cmClassesSet.contains(label) && cmClassesSet.contains(prediction)
    }

    val sortedThresholds = $(confMatrixThresholds).sorted.toIndexedSeq

    // reduce data to a coarser RDD (with size N * N * thresholds at most) for further aggregation
    val labelPredictionConfidenceCountRDD = dataTopNLabels.map{
      case (label, prediction, proba) => {
        ( (label, prediction, SearchHelper.findThreshold(sortedThresholds, proba.max)), 1L )
      }
    }.reduceByKey(_ + _)

    labelPredictionConfidenceCountRDD.persist()

    val cmByThreshold = sortedThresholds.map( threshold => {
      val filteredRDD = labelPredictionConfidenceCountRDD.filter {
        case ((_, _, confidence), _) => confidence >= threshold
      }
      constructConfusionMatrix(filteredRDD, cmClasses)
    })

    labelPredictionConfidenceCountRDD.unpersist()

    MulticlassConfMatrixMetricsByThreshold(
      ConfMatrixNumClasses = $(confMatrixNumClasses),
      ConfMatrixClassIndices = cmClasses,
      ConfMatrixThresholds = $(confMatrixThresholds),
      ConfMatrices = cmByThreshold
    )
  }

/**
 * function to calculate the mostly frequently mis-classified classes for each label/prediction category
 *
 * @param data RDD of (label, prediction)
 * @return a MisClassificationMetrics instance
 */
  def calculateMisClassificationMetrics(data: RDD[(Double, Double)]): MisClassificationMetrics = {

    val labelPredictionCountRDD = data.map {
      case (label, prediction) => ((label, prediction), 1L) }
      .reduceByKey(_ + _)

    val misClassificationsByLabel = labelPredictionCountRDD.map {
      case ((label, prediction), count) => (label, Seq((prediction, count)))
    }.reduceByKey(_ ++ _)
      .map { case (label, predictionCountsIter) => {
        val misClassificationCtMap = predictionCountsIter
          .filter { case (pred, _) => pred != label }
          .sortBy(-_._2)
          .take($(confMatrixMinSupport)).toMap

        val labelCount = predictionCountsIter.map(_._2).reduce(_ + _)
        val correctCount = predictionCountsIter
          .collect { case (pred, count) if pred == label => count }
          .reduceOption(_ + _).getOrElse(0L)

        MisClassificationsPerCategory(
          Category = label,
          TotalCount = labelCount,
          CorrectCount = correctCount,
          MisClassifications = misClassificationCtMap
        )
      }
    }.sortBy(-_.TotalCount).collect()

    val misClassificationsByPrediction = labelPredictionCountRDD.map {
      case ((label, prediction), count) => (prediction, Seq((label, count)))
    }.reduceByKey(_ ++ _)
      .map { case (prediction, labelCountsIter) => {
        val sortedMisclassificationCt = labelCountsIter
          .filter { case (label, _) => label != prediction }
          .sortBy(-_._2)
          .take($(confMatrixMinSupport)).toMap

        val predictionCount = labelCountsIter.map(_._2).reduce(_ + _)
        val correctCount = labelCountsIter
          .collect { case (label, count) if label == prediction => count }
          .reduceOption(_ + _).getOrElse(0L)

        MisClassificationsPerCategory(
          Category = prediction,
          TotalCount = predictionCount,
          CorrectCount = correctCount,
          MisClassifications = sortedMisclassificationCt
        )
      }
    }.sortBy(-_.TotalCount).collect()

    MisClassificationMetrics(
      ConfMatrixMinSupport = $(confMatrixMinSupport),
      MisClassificationsByLabel = misClassificationsByLabel,
      MisClassificationsByPrediction = misClassificationsByPrediction
    )
  }

/**
 * Function that calculates Multi Classification Metrics for different topK most occuring labels given an RDD
 * of scores & labels, and a list of topK values to consider.
 *
 * Output: MultiClassificationMetricsTopK object, containing an array of metrics of Precision, Recall, F1
 * and Error Rate for each of the topK values.
 *
 * @param data       Input RDD consisting of (vector of score probabilities, label), where label corresponds to the
 *                   index of the true class and the score vector consists of probabilities for each class
 * @param topKs      Sequence of topK values to calculate multiclass classification metrics for
 */
  def calculateTopKMetrics(
    data: RDD[(Double, Double)],
    topKs: Seq[Int]
  ): MultiClassificationMetricsTopK = {
    val labelCounts: RDD[(Double, Long)] = data.map { case (_, label) => (label, 1L)}.reduceByKey(_ + _)
    val sortedLabels: RDD[Double] = labelCounts.sortBy { case (_, count) => -1 * count}.map { case (label, _) => label}
    val topKLabels: Seq[Array[Double]] = topKs.map(k => sortedLabels.take(k))
    val topKMetrics: Seq[MulticlassMetrics] = topKLabels.map(topKLabel => {
      val filteredRdd: RDD[(Double, Double)] =
        data.map { case (pred, label) => if (topKLabel contains label) (pred, label) else (pred, -1) }
      new MulticlassMetrics(filteredRdd)
    })

    val error: Seq[Double] = topKMetrics.map(1.0 - _.accuracy)
    val precision: Seq[Double] = topKMetrics.map(_.weightedPrecision)
    val recall: Seq[Double] = topKMetrics.map(_.weightedRecall)
    val f1: Seq[Double] = (precision zip recall).map {
      case (precision, recall) => if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)
    }

    MultiClassificationMetricsTopK(
      topKs = topKs,
      Precision = precision,
      Recall = recall,
      F1 = f1,
      Error = error
    )
  }

  /**
   * Function that calculates a set of threshold metrics for different topN values given an RDD of scores & labels,
   * a list of topN values to consider, and a list of thresholds to use.
   *
   * Output: ThresholdMetrics object, containing thresholds used, topN values used, and maps from topN value to
   * arrays of correct, incorrect, and no prediction counts at each threshold. Summing all three of these arrays
   * together should give an array where each entry the total number of rows in the input RDD.
   *
   * @param data       Input RDD consisting of (vector of score probabilities, label), where label corresponds to the
   *                   index of the true class and the score vector consists of probabilities for each class
   * @param topNs      Sequence of topN values to calculate threshold metrics for.
   *                   For example, if topN is Seq(1, 3, 10) then threshold metrics are calculated by considering if
   *                   the score of the true class is in the top 1, top 3, and top10 scores, respectively. If a topN
   *                   value is greater than the number of total classes,
   *                   then it will still be applied, but will have the same results as if that topN value = num classes
   * @param thresholds Sequence of threshold values applied to predicted probabilities, therefore they must be in the
   *                   range [0.0, 1.0]
   */
  def calculateThresholdMetrics(
    data: RDD[(Array[Double], Double)],
    topNs: Seq[Int],
    thresholds: Seq[Double]
  ): MulticlassThresholdMetrics = {
    require(thresholds.nonEmpty, "thresholds sequence in cannot be empty")
    require(thresholds.forall(x => x >= 0 && x <= 1.0), "thresholds sequence elements must be in the range [0, 1]")
    require(topNs.nonEmpty, "topN sequence in cannot be empty")
    require(topNs.forall(_ > 0), "topN sequence can only contain positive integers")

    type Label = Int
    type CorrIncorr = (Array[Long], Array[Long])
    type MetricsMap = Map[Label, CorrIncorr]

    val nThresholds = thresholds.length

    /**
     * Allocates an array of longs and fills it with a specified value from start until end
     */
    def arrayFill(size: Int)(start: Int, end: Int, value: Long) = {
      val res = new Array[Long](size)
      var i = start
      while (i < end) {
        res(i) = value
        i += 1
      }
      res
    }

    /**
     * First aggregation step turns an array of scores (as probabilities) and a single label (index of correct class)
     * into two arrays, correct and incorrect counts by threshold. Each array index corresponds to whether
     * the score counts as correct or incorrect at the threshold corresponding to that index.
     */
    def computeMetrics(scoresAndLabels: (Array[Double], Double)): MetricsMap = {
      val scores: Array[Double] = scoresAndLabels._1
      val label: Label = scoresAndLabels._2.toInt
      // The label may be unseen during model training, so treat scores for unseen classes as all being zero
      val trueClassScore: Double = scores.lift(label).getOrElse(0.0)
      val topNsAndScores: Map[Label, Array[(Double, Int)]] = topNs.map(t => t -> scores.zipWithIndex.sortBy(-_._1)
        .take(t)).toMap
      val topNScores: Map[Label, Array[Double]] = topNsAndScores.mapValues(_.map(_._1))
      // Doesn't matter which key you use since the scores are sorted
      val topScore: Double = topNScores.head._2.head
      val topNIndices: Map[Label, Array[Int]] = topNsAndScores.mapValues(_.map(_._2))

      // To calculate correct / incorrect counts per threshold, we just need to find the array index where the
      // true label score and the top score are no longer >= threshold.
      val trueScoreCutoffIndex: Int = {
        val idx = thresholds.indexWhere(_ > trueClassScore)
        if (idx < 0) nThresholds else idx
      }
      val maxScoreCutoffIndex: Int = {
        val idx = thresholds.indexWhere(_ > topScore)
        if (idx < 0) nThresholds else idx
      }
      topNs.view.map { t =>
        val correctCounts = if (topNIndices(t).contains(label)) {
          arrayFill(nThresholds)(start = 0, end = trueScoreCutoffIndex, value = 1L)
        } else new Array[Long](nThresholds)

        val incorrectCounts = if (topNIndices(t).contains(label)) {
          arrayFill(nThresholds)(start = trueScoreCutoffIndex, end = maxScoreCutoffIndex, value = 1L)
        } else arrayFill(nThresholds)(start = 0, end = maxScoreCutoffIndex, value = 1L)

        t -> (correctCounts, incorrectCounts)
      }.toMap[Label, CorrIncorr]
    }

    val zeroValue: MetricsMap =
      topNs
        .map(_ -> (new Array[Long](nThresholds), new Array[Long](nThresholds)))
        .toMap[Label, CorrIncorr]

    implicit val sgTuple2 = new Tuple2Semigroup[Array[Long], Array[Long]]()
    val agg: MetricsMap = data.treeAggregate[MetricsMap](zeroValue)(combOp = _ + _, seqOp = _ + computeMetrics(_))

    val nRows = data.count()
    MulticlassThresholdMetrics(
      topNs = topNs,
      thresholds = thresholds,
      correctCounts = agg.map { case (k, (cor, _)) => k -> cor.toSeq },
      incorrectCounts = agg.map { case (k, (_, incor)) => k -> incor.toSeq },
      noPredictionCounts = agg.map { case (k, (cor, incor)) =>
        k -> (Array.fill(nThresholds)(nRows) + cor.map(-_) + incor.map(-_)).toSeq
      }
    )
  }

  final protected def getMultiEvaluatorMetric(
    metricName: ClassificationEvalMetric,
    dataset: Dataset[_],
    default: => Double
  ): Double = {
    val labelName = getLabelCol
    val dataUse = makeDataToUse(dataset, labelName)
    new MulticlassClassificationEvaluator()
      .setLabelCol(labelName)
      .setPredictionCol(getPredictionValueCol)
      .setMetricName(metricName.sparkEntryName)
      .evaluateOrDefault(dataUse, default = default)
  }
}


/**
 * Metrics of MultiClassification Problem
 *
 * @param Precision
 * @param Recall
 * @param F1
 * @param Error
 * @param ThresholdMetrics
 */
case class MultiClassificationMetrics
(
  Precision: Double,
  Recall: Double,
  F1: Double,
  Error: Double,
  ThresholdMetrics: MulticlassThresholdMetrics,
  TopKMetrics: MultiClassificationMetricsTopK,
  ConfusionMatrixMetrics: MulticlassConfMatrixMetricsByThreshold,
  MisClassificationMetrics: MisClassificationMetrics
) extends EvaluationMetrics

/**
 * Metrics for topK MultiClassification
 *
 * Each metric contains a list of metrics corresponding to each of the topK most occurring labels.  If the predicted
 * label is outside of the topK most occurring labels, it is treated as incorrect.
 *
 * @param topKs
 * @param Precision
 * @param Recall
 * @param F1
 * @param Error
 */
case class MultiClassificationMetricsTopK
(
  topKs: Seq[Int],
  Precision: Seq[Double],
  Recall: Seq[Double],
  F1: Seq[Double],
  Error: Seq[Double]
) extends EvaluationMetrics

/**
 * Metrics for multi-class confusion matrix. It captures confusion matrix of records of which
 * 1) the labels belong to the top n most occurring classes (n = confMatrixNumClasses)
 * 2) the top predicted probability exceeds a certain threshold in confMatrixThresholds
 *
 * @param confMatrixNumClasses value of the top n most occurring classes in the dataset
 * @param confMatrixClassIndices label index of the top n most occuring classes
 * @param confMatrixThresholds a sequence of thresholds
 * @param confMatrices a sequence of counts that stores the confusion matrix for each threshold in confMatrixThresholds
 */
case class MulticlassConfMatrixMetricsByThreshold
(
  ConfMatrixNumClasses: Int,
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  ConfMatrixClassIndices: Seq[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  ConfMatrixThresholds: Seq[Double],
  ConfMatrices: Seq[Seq[Long]]
) extends EvaluationMetrics

/**
 * Multiclass mis-classification metrics, including the top n (n = confMatrixMinSupport) most frequently
 * mis-classified classes for each label or prediction category.
 *
 */
case class MisClassificationMetrics
(
  ConfMatrixMinSupport: Int,
  MisClassificationsByLabel: Seq[MisClassificationsPerCategory],
  MisClassificationsByPrediction: Seq[MisClassificationsPerCategory]
)

/**
 * Case class that stores the most frequently mis-classified classes for each label/prediction category
 *
 * @param category a category which a record's label or prediction equals to
 * @param totalCount total # of records in that category
 * @param correctCount # of correctly predicted records in that category
 * @param misClassifications the top n most frequently misclassified classes (n = confMatrixMinSupport) and
 *                           their respective counts in that category. Ordered by counts in descending order.
 */
case class MisClassificationsPerCategory
(
  Category: Double,
  TotalCount: Long,
  CorrectCount: Long,
  @JsonDeserialize(keyAs = classOf[java.lang.Double])
  MisClassifications: Map[Double, Long]
)

case class labelPredictionConfidence
(
  Label: Double,
  Prediction: Double,
  Confidence: Double
)

/**
 * Threshold-based metrics for multiclass classification
 *
 * Classifications being correct, incorrect, or no classification are defined in terms of the topN and score threshold
 * to be:
 * Correct - score of the true label is in the top N scores AND the score of the true label is >= threshold
 * Incorrect - score of top predicted label >= threshold AND
 * (true label NOT in top N predicted labels OR score of true label < threshold)
 * No prediction - otherwise (score of top predicted label < threshold)
 *
 * @param topNs              list of topN values (used as keys for the count maps)
 * @param thresholds         list of threshold values (correspond to thresholds at the indices
 *                           of the arrays in the count maps)
 * @param correctCounts      map from topN value to an array of counts of correct classifications at each threshold
 * @param incorrectCounts    map from topN value to an array of counts of incorrect classifications at each threshold
 * @param noPredictionCounts map from topN value to an array of counts of no prediction at each threshold
 */
case class MulticlassThresholdMetrics
(
  @JsonDeserialize(contentAs = classOf[java.lang.Integer])
  topNs: Seq[Int],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  thresholds: Seq[Double],
  @JsonDeserialize(keyAs = classOf[java.lang.Integer])
  correctCounts: Map[Int, Seq[Long]],
  @JsonDeserialize(keyAs = classOf[java.lang.Integer])
  incorrectCounts: Map[Int, Seq[Long]],
  @JsonDeserialize(keyAs = classOf[java.lang.Integer])
  noPredictionCounts: Map[Int, Seq[Long]]
) extends EvaluationMetrics

