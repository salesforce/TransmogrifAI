package org.apache.spark.mllib.evaluation

import org.apache.spark.mllib.evaluation.binary.BinaryConfusionMatrix
import org.apache.spark.rdd.RDD

class ExtendedBinaryClassificationMetrics(
  override val scoreAndLabels: RDD[(Double, Double)],
  override val numBins: Int
) extends BinaryClassificationMetrics(scoreAndLabels, numBins) {

  /**
   * Exposes the thresholded confusion matrices that Spark uses to calculate other derived metrics.
   * @return RDD of (threshold, BinaryConfusionMatrix) over all thresholds calculated
   */
  def confusionMatrixByThreshold(): RDD[(Double, BinaryConfusionMatrix)] = {
    val method = this.getClass.getSuperclass.getDeclaredMethod("confusions")
    method.setAccessible(true)
    method.invoke(this).asInstanceOf[RDD[(Double, BinaryConfusionMatrix)]]
  }
}
