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

package com.salesforce.op.stages.impl.preparators

import com.salesforce.op.UID
import com.salesforce.op.features.types.OPVector
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.slf4j.impl.Log4jLoggerAdapter


/**
 * The MinVarianceFilter checks that computed features have a minimum variance
 *
 * Like SanityChecker, the Estimator step outputs statistics on incoming data, as well as the
 * names of features which should be dropped from the feature vector. And the transformer step
 * applies the action of actually removing the low variance features from the feature vector
 *
 * Two distinctions from SanityChecker:
 * (1) no label column as input; and
 * (2) only filters features by variance
 */
class MinVarianceFilter
(
  operationName: String = classOf[MinVarianceFilter].getSimpleName,
  uid: String = UID[MinVarianceFilter]
) extends UnaryEstimator[OPVector, OPVector](operationName = operationName, uid = uid)
  with DerivedFeatureFilterParams {

  setDefault(
    removeBadFeatures -> MinVarianceFilter.RemoveBadFeatures,
    minVariance -> MinVarianceFilter.MinVariance
  )

  override def fitFn(data: Dataset[OPVector#Value]): UnaryModel[OPVector, OPVector] = {
    // Set the desired log level
    if (isSet(logLevel)) {
      Option(log).collect { case l: Log4jLoggerAdapter =>
        LogManager.getLogger(l.getName).setLevel(Level.toLevel($(logLevel)))
      }
    }
    val removeBad = $(removeBadFeatures)

    logDebug("Getting vector rows")
    val vectorRows: RDD[OldVector] = data.rdd.map {
      case sparse: SparseVector => OldVectors.sparse(sparse.size, sparse.indices, sparse.values)
      case dense: DenseVector => OldVectors.dense(dense.toArray)
    }.persist()

    logDebug("Calculating columns stats")
    val colStats = Statistics.colStats(vectorRows)
    val count = colStats.count
    require(count > 0, "Sample size cannot be zero")

    val featureSize = vectorRows.first().size
    require(featureSize > 0, "Feature vector passed in is empty, check your vectorizers")

    // handle any possible serialization errors if users give us wrong metadata
    val vectorMeta = try {
      OpVectorMetadata(getInputSchema()(in1.name))
    } catch {
      case e: NoSuchElementException =>
        throw new IllegalArgumentException("Vector input metadata is malformed: ", e)
    }

    require(featureSize == vectorMeta.size,
      s"Number of columns in vector metadata (${vectorMeta.size}) did not match number of columns in data" +
        s"($featureSize), check your vectorizers \n metadata=$vectorMeta")
    val vectorMetaColumns = vectorMeta.columns
    val featureNames = vectorMetaColumns.map(_.makeColName())

    logDebug("Logging all statistics")
    val stats = DerivedFeatureFilterUtils.makeColumnStatistics(vectorMetaColumns, colStats)
    stats.foreach { stat => logDebug(stat.toString) }

    logDebug("Calculating features to remove")
    val (toDropFeatures, warnings) = if (removeBad) {
      DerivedFeatureFilterUtils.getFeaturesToDrop(stats, $(minVariance)).unzip
    } else (Array.empty[ColumnStatistics], Array.empty[String])
    warnings.foreach { warning => logWarning(warning) }

    val toDropSet = toDropFeatures.flatMap(_.column).toSet
    val outputFeatures = vectorMetaColumns.filterNot { col => toDropSet.contains(col) }
    val indicesToKeep = outputFeatures.map(_.index)

    val outputMeta = OpVectorMetadata(getOutputFeatureName, outputFeatures, vectorMeta.history)

    val summary = new MinVarianceSummary(
      dropped = toDropFeatures.map(_.name),
      featuresStatistics = new SummaryStatistics(colStats, sample = 1.0),
      names = featureNames
    )

    setMetadata(outputMeta.toMetadata.withSummaryMetadata(summary.toMetadata()))

    vectorRows.unpersist(blocking = false)

    require(indicesToKeep.length > 0,
      "The minimum variance filter has dropped all of your features, check your input data or your threshold")

    new MinVarianceFilterModel(
      indicesToKeep = indicesToKeep,
      removeBadFeatures = removeBad,
      operationName = operationName,
      uid = uid
    )
  }
}

final class MinVarianceFilterModel private[op]
(
  val indicesToKeep: Array[Int],
  val removeBadFeatures: Boolean,
  operationName: String,
  uid: String
) extends UnaryModel[OPVector, OPVector](operationName = operationName, uid = uid) {

  def transformFn: OPVector => OPVector = DerivedFeatureFilterUtils.removeFeatures(indicesToKeep, removeBadFeatures)
}

object MinVarianceFilter {
  val RemoveBadFeatures = false
  val MinVariance = 1E-5
}
