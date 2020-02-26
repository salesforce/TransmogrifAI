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
import com.salesforce.op.features.types.{OPVector, VectorConversions}
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.log4j.{Level, LogManager}
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors => NewVectors}
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, Param, Params}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.MetadataBuilder
import org.slf4j.impl.Log4jLoggerAdapter

import scala.util.Try


trait MinVarianceFilterParams extends Params {

  final val logLevel = new Param[String](
    parent = this, name = "logLevel",
    doc = "sets log level (INFO, WARN, ERROR, DEBUG etc.)",
    isValid = (s: String) => Try(Level.toLevel(s)).isSuccess
  )

  private[op] def setLogLevel(level: Level): this.type = set(logLevel, level.toString)

  final val removeBadFeatures = new BooleanParam(
    parent = this, name = "removeBadFeatures",
    doc = "If set to true, this will automatically remove all the bad features from the feature vector"
  )

  def setRemoveBadFeatures(value: Boolean): this.type = set(removeBadFeatures, value)

  def getRemoveBadFeatures: Boolean = $(removeBadFeatures)

  final val minVariance = new DoubleParam(
    parent = this, name = "minVariance",
    doc = "Minimum amount of variance allowed for each feature"
  )

  def setMinVariance(value: Double): this.type = set(minVariance, value)

  def getMinVariance: Double = $(minVariance)

  setDefault(
    removeBadFeatures -> MinVarianceFilter.RemoveBadFeatures,
    minVariance -> MinVarianceFilter.MinVariance
  )
}

class MinVarianceFilter
(
  operationName: String = classOf[MinVarianceFilter].getSimpleName,
  uid: String = UID[MinVarianceFilter]
) extends UnaryEstimator[OPVector, OPVector](operationName = operationName, uid = uid)
  with MinVarianceFilterParams {

  private def makeColumnStatistics
  (
    metaCols: Seq[OpVectorColumnMetadata],
    statsSummary: MultivariateStatisticalSummary
  ): Array[ColumnStatistics] = {
    val means = statsSummary.mean
    val maxs = statsSummary.max
    val mins = statsSummary.min
    val count = statsSummary.count
    val variances = statsSummary.variance
    val featuresStats = metaCols.map {
      col =>
        val i = col.index
        val name = col.makeColName()
        ColumnStatistics(
          name = name,
          column = Some(col),
          isLabel = false,
          count = count,
          mean = means(i),
          min = mins(i),
          max = maxs(i),
          variance = variances(i),
          corrLabel = None,
          cramersV = None,
          parentCorr = None,
          parentCramersV = None,
          maxRuleConfidences = Seq.empty,
          supports = Seq.empty
        )
    }
    featuresStats.toArray
  }

  private def getFeaturesToDrop(stats: Array[ColumnStatistics]): Array[ColumnStatistics] = {
    val minVar = $(minVariance)
    // Including dummy params in reasonsToRemove to allow re-use of `ColumnStatistics`
    for {
      col <- stats
      reasons = col.reasonsToRemove(
        minVariance = minVar,
        minCorrelation = 0.0,
        maxCorrelation = 1.0,
        maxCramersV = 1.0,
        maxRuleConfidence = 1.0,
        minRequiredRuleSupport = 1.0,
        removeFeatureGroup = false,
        protectTextSharedHash = true,
        removedGroups = Seq.empty
      )
      if reasons.nonEmpty
    } yield {
      logWarning(s"Removing ${col.name} due to: ${reasons.mkString(",")}")
      col
    }
  }

  override def fitFn(data: Dataset[OPVector#Value]): UnaryModel[OPVector, OPVector] = {
    // Set the desired log level
    if (isSet(logLevel)) {
      Option(log).collect { case l: Log4jLoggerAdapter =>
        LogManager.getLogger(l.getName).setLevel(Level.toLevel($(logLevel)))
      }
    }
    val removeBad = $(removeBadFeatures)

    logInfo("Getting vector rows")
    val vectorRows: RDD[OldVector] = data.rdd.map {
      case sparse: SparseVector => OldVectors.sparse(sparse.size, sparse.indices, sparse.values)
      case dense: DenseVector => OldVectors.dense(dense.toArray)
    }.persist()

    logInfo("Calculating columns stats")
    val colStats = Statistics.colStats(vectorRows)
    val count = colStats.count
    require(count > 0, "Sample size cannot be zero")

    val featureSize = vectorRows.first().size - 1
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

    logInfo("Logging all statistics")
    val stats = makeColumnStatistics(vectorMetaColumns, colStats)
    stats.foreach { stat => logInfo(stat.toString) }

    logInfo("Calculating features to remove")
    val toDropFeatures = if (removeBad) getFeaturesToDrop(stats) else Array.empty[ColumnStatistics]
    val toDropSet = toDropFeatures.flatMap(_.column).toSet
    val outputFeatures = vectorMetaColumns.filterNot { col => toDropSet.contains(col) }
    val indicesToKeep = outputFeatures.map(_.index)

    val outputMeta = OpVectorMetadata(getOutputFeatureName, outputFeatures, vectorMeta.history)

    val summaryMetadata = {
      val featuresStatistics = new SummaryStatistics(colStats, sample = 1.0)
      val summaryMeta = new MetadataBuilder()
      summaryMeta.putStringArray(SanityCheckerNames.Dropped, toDropFeatures.map(_.name))
      summaryMeta.putMetadata(SanityCheckerNames.FeaturesStatistics, featuresStatistics.toMetadata())
      summaryMeta.putStringArray(SanityCheckerNames.Names, featureNames)
      summaryMeta.build()
    }

    setMetadata(outputMeta.toMetadata.withSummaryMetadata(summaryMetadata))

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

  def transformFn: OPVector => OPVector = feature => {
    if (!removeBadFeatures) feature
    else {
      val vals = new Array[Double](indicesToKeep.length)
      feature.value.foreachActive((i, v) => {
        val k = indicesToKeep.indexOf(i)
        if (k >= 0) vals(k) = v
      })
      NewVectors.dense(vals).compressed.toOPVector
    }
  }
}

object MinVarianceFilter {
  val RemoveBadFeatures = false
  val MinVariance = 1E-5
}
